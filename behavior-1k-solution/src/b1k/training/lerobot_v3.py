"""LeRobot v3 (2026-challenge-demos) dataset support for the PI_BEHAVIOR training pipeline.

The original pipeline reads the 2025 challenge data through OmniGibson's ``BehaviorLeRobotDataset``
(2025 OmniGibson, v2.1-style per-task layout). The 2026 data ships as a LeRobot v3.0 dataset
(``behavior-1k/2026-challenge-demos``: 100 tasks, ``observation.state`` 61-dim, camera keys
``observation.rgb.<camera>_camera_0``), and users typically pull a per-task subset of it into a local
root. This module loads such a root with the LeRobot v3 loader and adapts the samples:

* ``RepackFirstAvailable`` maps the several camera/state key spellings seen in the wild onto the
  ``observation/...`` names the B1K transforms expect.
* ``resolve_task_id_map`` maps the dataset's local ``task_index`` to the 2026 challenge task id
  (== model task-embedding row), from task names / instructions in ``meta/tasks``.
* ``ComputeSubtaskStateV3`` computes the timestamp-based stage label using episode lengths taken
  from the data parquets (see ``episode_files.v3_episode_lengths``).
"""
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from openpi.transforms import DataDict, DataTransformFn, flatten_dict

from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.shared.task_list_2026 import INSTRUCTIONS, TASK_NAME_TO_ID
from b1k.training.episode_files import v3_episode_lengths

logger = logging.getLogger(__name__)

# Candidate source keys for each B1K input, first match wins. Covers the official 2026 export
# (observation.rgb.<cam>_camera_0), the 2025-style names (observation.images.rgb.<role>) used by
# some local re-exports, and the resized 224 export used by the original solution.
REPACK_CANDIDATES_2026: dict[str, Sequence[str]] = {
    "observation/egocentric_camera": (
        "observation.rgb.zed_link_camera_0",
        "observation.images.rgb.head",
        "observation.images.head",
    ),
    "observation/wrist_image_left": (
        "observation.rgb.left_realsense_link_camera_0",
        "observation.images.rgb.left_wrist",
        "observation.images.left_wrist",
    ),
    "observation/wrist_image_right": (
        "observation.rgb.right_realsense_link_camera_0",
        "observation.images.rgb.right_wrist",
        "observation.images.right_wrist",
    ),
    "observation/state": ("observation.state",),
    "actions": ("action",),
    "task_index": ("task_index",),
    "timestamp": ("timestamp",),
    "episode_index": ("episode_index",),
    "index": ("index",),
}


@dataclasses.dataclass(frozen=True)
class RepackFirstAvailable(DataTransformFn):
    """Like openpi's RepackTransform, but each target takes the first candidate key present."""

    candidates: dict[str, Sequence[str]]

    def __call__(self, data: DataDict) -> DataDict:
        flat = flatten_dict(data)
        out = {}
        for new_key, options in self.candidates.items():
            for old_key in options:
                if old_key in flat:
                    out[new_key] = flat[old_key]
                    break
            else:
                raise KeyError(f"None of {list(options)} found in sample for '{new_key}'. Keys: {sorted(flat)[:40]}")
        return out


def resolve_task_id_map(tasks: dict[int, str], explicit: dict[int, int] | None = None) -> dict[int, int]:
    """Map dataset task_index -> 2026 challenge task id.

    ``tasks`` is {task_index: task_string}; the string is a task name in the official export, or a
    natural-language instruction in some subsets. ``explicit`` overrides/complements the automatic
    mapping. Raises if any task_index stays unresolved (a wrong embedding row would silently train the
    wrong task).
    """
    out: dict[int, int] = dict(explicit or {})
    unresolved = []
    for task_index, task_str in tasks.items():
        task_index = int(task_index)
        if task_index in out:
            continue
        if task_str in TASK_NAME_TO_ID:
            out[task_index] = TASK_NAME_TO_ID[task_str]
        elif task_str in INSTRUCTIONS:
            out[task_index] = INSTRUCTIONS[task_str]
        else:
            unresolved.append((task_index, task_str))
    if unresolved:
        raise ValueError(
            "Could not map these dataset tasks to 2026 challenge task ids; pass task_id_map={task_index: task_id} "
            f"in the data config: {unresolved}"
        )
    return out


def tasks_from_meta(meta) -> dict[int, str]:
    """{task_index: task_string} from a LeRobotDatasetMetadata (v3: DataFrame indexed by task string)."""
    tasks = meta.tasks
    if hasattr(tasks, "itertuples"):  # pandas DataFrame, index = task string, column task_index
        return {int(row.task_index): str(idx) for idx, row in zip(tasks.index, tasks.itertuples(index=False))}
    if isinstance(tasks, dict):
        return {int(k): str(v) for k, v in tasks.items()}
    raise TypeError(f"Unrecognised tasks metadata type {type(tasks)}")


@dataclasses.dataclass(frozen=True)
class ComputeSubtaskStateV3(DataTransformFn):
    """Timestamp-based stage label (same rule as ComputeSubtaskStateFromMeta) for LeRobot v3 samples.

    Uses the mapped 2026 task id for the per-task stage count and episode lengths keyed by the
    ``episode_index`` value the sample carries.
    """

    episode_lengths: dict[int, int]
    task_id_map: dict[int, int]
    fps: float = 30.0

    def __call__(self, data: DataDict) -> DataDict:
        task_id = self.task_id_map[int(data["task_index"])]
        num_stages = TASK_NUM_STAGES[task_id]
        episode_length = self.episode_lengths[int(data["episode_index"])]
        current_step = float(data["timestamp"]) * self.fps
        stage = int(current_step / (episode_length / num_stages))
        data["subtask_state"] = np.array(max(0, min(stage, num_stages - 1)), dtype=np.int32)
        return data


def _skill_span(frame_duration) -> tuple[int, int]:
    """(start, end) of a skill. Some 2026 annotations (task 7) split one skill into segments [[a, b], [c, d]]:
    span = first segment start .. last segment end."""
    fd = frame_duration
    if fd and isinstance(fd[0], (list, tuple)):
        return int(fd[0][0]), int(fd[-1][1])
    return int(fd[0]), int(fd[1])


def load_skill_ends(data_root: str | Path, episode_lengths: dict[int, int], max_len_diff: float = 0.10) -> dict[int, tuple[int, ...]]:
    """Sorted skill END frames per episode, from the demos' human skill annotations.

    ``<root>/annotations/task-NNNN/episode_*.json`` (pulled with the dataset) carries one
    ``skill_annotation`` list per demo ("move to", "open door", "pick up from", ... with frame
    ranges). The files are not keyed by ``episode_index``; both sides are in the same order, so they
    are paired by sorted order and each pair must agree on length (annotation ``task_duration``
    within ``max_len_diff`` of the episode length, relative, min 20 frames). Annotations are often a
    little shorter than the recording (trimmed tail): task 51 within 7 frames, task 49 up to 3.4%, task 34
    up to 7.8%, never by much and nearly always shorter, which a mis-pairing would not look like. Episodes whose pairing or labels look
    wrong (first skill starting after 20% of the demo, last one ending before 80%, < 2 skills) are
    left out, and the caller falls back to the time rule for them.
    """
    import json

    files = sorted(Path(data_root).glob("annotations/task-*/episode_*.json"))
    episodes = sorted(episode_lengths)
    if len(files) != len(episodes):
        logger.warning(f"skill annotations: {len(files)} files for {len(episodes)} episodes; pairing by order anyway")
    out: dict[int, tuple[int, ...]] = {}
    for ep, f in zip(episodes, files):
        d = json.loads(f.read_text())
        length = episode_lengths[ep]
        skills = d.get("skill_annotation") or []
        if abs(length - int(d["meta_data"]["task_duration"])) > max(20, max_len_diff * length) or len(skills) < 2:
            continue
        starts = [_skill_span(s["frame_duration"])[0] for s in skills]
        ends = sorted(_skill_span(s["frame_duration"])[1] for s in skills)
        if min(starts) > 0.2 * length or ends[-1] < 0.8 * length:
            continue
        out[ep] = tuple(ends)
    if not out:
        logger.warning(f"skill annotations: NONE usable under {data_root}/annotations; every episode uses time stages")
    else:
        logger.info(f"skill annotations: usable for {len(out)}/{len(episodes)} episodes")
    return out


class _RepeatFrames:
    """Index view over a dataset that lists some frames more than once (oversampling). Other attributes pass through."""

    def __init__(self, base, index):
        self._base = base
        self._index = index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        return self._base[int(self._index[i])]

    def __getattr__(self, name):
        # Same guard as below: DataLoader workers unpickle before __dict__ is restored -> no recursion.
        if name in ("_base", "_index") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._base, name)


def oversample_skill_frames(dataset, data_root: str | Path, skills: tuple[str, ...], factor: int):
    """Repeat every frame inside the named skill-annotation segments ``factor`` times in total.

    Annotations are paired with episodes the same way as ``load_skill_ends`` (sorted order, length check).
    """
    import json

    from b1k.training.episode_files import v3_episode_lengths

    lengths = v3_episode_lengths(data_root)
    files = sorted(Path(data_root).glob("annotations/task-*/episode_*.json"))
    windows: dict[int, list[tuple[int, int]]] = {}
    for ep, f in zip(sorted(lengths), files):
        d = json.loads(f.read_text())
        if abs(lengths[ep] - int(d["meta_data"]["task_duration"])) > max(20, 0.10 * lengths[ep]):
            continue
        w = [_skill_span(s["frame_duration"]) for s in d.get("skill_annotation") or []
             if s["skill_description"] and s["skill_description"][0] in skills]
        if w:
            windows[ep] = w
    cols = dataset.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")
    ep_arr, fr_arr = np.asarray(cols["episode_index"]), np.asarray(cols["frame_index"])
    assert len(ep_arr) == len(dataset), f"hf_dataset rows {len(ep_arr)} != dataset length {len(dataset)}"
    hit = np.zeros(len(ep_arr), dtype=bool)
    for ep, w in windows.items():
        rows = np.flatnonzero(ep_arr == ep)
        inside = np.zeros(len(rows), dtype=bool)
        for a, b in w:
            inside |= (fr_arr[rows] >= a) & (fr_arr[rows] <= b)
        hit[rows[inside]] = True
    index = np.concatenate([np.arange(len(dataset)), np.repeat(np.flatnonzero(hit), factor - 1)])
    logger.info(f"oversample {skills} x{factor}: {int(hit.sum())}/{len(hit)} frames in {len(windows)} episodes "
                f"-> {len(index)} samples per epoch")
    return _RepeatFrames(dataset, index)


@dataclasses.dataclass(frozen=True)
class ComputeSubtaskStateFromSkills(DataTransformFn):
    """Stage label from task progress instead of elapsed time.

    stage = (skills finished so far) * num_stages // (skills in this demo), so stage boundaries fall on
    real step boundaries ("brisket is out of the fridge") and keep the task's order, rather than on
    equal time slices. Overlapping (bimanual) skills are fine: only end frames are counted. Episodes
    without usable annotations use the time rule of ComputeSubtaskStateV3. The stage count per task is
    unchanged, so checkpoints trained with the time rule load as-is.
    """

    episode_lengths: dict[int, int]
    task_id_map: dict[int, int]
    skill_ends: dict[int, tuple[int, ...]]
    fps: float = 30.0

    def __call__(self, data: DataDict) -> DataDict:
        task_id = self.task_id_map[int(data["task_index"])]
        num_stages = TASK_NUM_STAGES[task_id]
        episode = int(data["episode_index"])
        current_step = float(data["timestamp"]) * self.fps
        ends = self.skill_ends.get(episode)
        if ends is None:
            stage = int(current_step / (self.episode_lengths[episode] / num_stages))
        else:
            stage = int(np.searchsorted(ends, current_step, side="right")) * num_stages // len(ends)
        data["subtask_state"] = np.array(max(0, min(stage, num_stages - 1)), dtype=np.int32)
        return data


class _EpisodesByValue:
    """View over a LeRobot ``meta.episodes`` table that indexes by ``episode_index`` value.

    LeRobot v3 assumes ``episode_index`` values are 0..N-1 and uses them as row positions into
    ``meta.episodes``. A per-task subset pulled out of the 100-task export keeps the original values
    (e.g. task 89 = 17800..17999) so positional lookups fail. This view keeps the table intact and maps
    a value to its row; negative ints and slices stay positional; everything else is delegated.
    """

    def __init__(self, table):
        self._table = table
        self._pos = {int(v): i for i, v in enumerate(table["episode_index"])}

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            k = int(key)
            if k < 0:
                return self._table[k]
            if k in self._pos:
                return self._table[self._pos[k]]
            raise KeyError(f"episode_index {k} not in this dataset")
        return self._table[key]

    def __len__(self):
        return len(self._table)

    def __iter__(self):
        return iter(self._table)

    def __getattr__(self, name):
        # Only reached for names not found normally. Guard the wrapped attributes themselves so that
        # unpickling in DataLoader worker processes (which calls __getattr__ before __dict__ is
        # restored) cannot recurse.
        if name in ("_table", "_pos") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._table, name)

    def __getstate__(self):
        return {"_table": self._table, "_pos": self._pos}

    def __setstate__(self, state):
        self.__dict__.update(state)


_LEROBOT_PATCHED = False


def patch_lerobot_for_subsets() -> None:
    """Make the LeRobot v3 loader accept subsets whose episode_index values are not 0..N-1."""
    global _LEROBOT_PATCHED
    if _LEROBOT_PATCHED:
        return
    from lerobot.datasets import dataset_metadata as dm

    load_name = "_load_metadata" if hasattr(dm.LeRobotDatasetMetadata, "_load_metadata") else "load_metadata"
    orig_load = getattr(dm.LeRobotDatasetMetadata, load_name)

    def load_metadata(self):
        orig_load(self)
        eps = self.episodes
        if eps is not None and not isinstance(eps, _EpisodesByValue):
            values = [int(v) for v in eps["episode_index"]]
            if values != list(range(len(values))):
                self.episodes = _EpisodesByValue(eps)
                logger.info(f"episode_index values are not 0..N-1 ({values[0]}..{values[-1]}); indexing meta.episodes by value")

    def _file_path(kind):
        def fn(self, ep_index, *args):
            if self.episodes is None:
                self.episodes = dm.load_episodes(self.root)
            ep = self.episodes[ep_index]
            if kind == "video":
                (vid_key,) = args
                return Path(self.video_path.format(video_key=vid_key, chunk_index=ep[f"videos/{vid_key}/chunk_index"], file_index=ep[f"videos/{vid_key}/file_index"]))
            return Path(self.data_path.format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
        return fn

    setattr(dm.LeRobotDatasetMetadata, load_name, load_metadata)
    dm.LeRobotDatasetMetadata.get_video_file_path = _file_path("video")
    dm.LeRobotDatasetMetadata.get_data_file_path = _file_path("data")
    _LEROBOT_PATCHED = True


def create_lerobot_v3_dataset(
    root: str | Path,
    repo_id: str,
    action_horizon: int,
    *,
    episodes: list[int] | None = None,
    tolerance_s: float = 0.5,
    video_backend: str | None = None,
    action_key: str = "action",
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    patch_lerobot_for_subsets()
    root = Path(root)
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if episodes is None:
        # Explicit list of episode_index VALUES: the loader's cache check otherwise assumes 0..N-1.
        episodes = [int(v) for v in meta.episodes["episode_index"]]
    t = time.time()
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps={action_key: [i / meta.fps for i in range(action_horizon)]},
        tolerance_s=tolerance_s,
        video_backend=video_backend,
    )
    logger.info(
        f"Loaded LeRobot v3 dataset {repo_id} from {root}: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames ({time.time() - t:.1f}s)"
    )
    return dataset, meta


# Apply at import time, not only in create_lerobot_v3_dataset(): DataLoader workers are spawned
# processes that re-import modules, and they reach this module when they unpickle the dataset
# (its metadata holds an _EpisodesByValue), so the class patches must be installed here too.
patch_lerobot_for_subsets()
