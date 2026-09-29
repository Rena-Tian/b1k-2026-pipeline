"""Enumerate and read per-episode (state, action) tables from either B1K dataset layout.

* 2025 layout (``behavior-1k/2025-challenge-demos`` / ``IliaLarchenko/behavior_224_rgb``):
  one parquet per episode at ``data/task-*/episode_*.parquet``.
* LeRobot v3 layout (``behavior-1k/2026-challenge-demos`` and subsets pulled from it):
  multi-episode parquets at ``data/chunk-*/file-*.parquet``; ``meta/episodes/**/*.parquet`` says
  which file each episode lives in.

Used by scripts/compute_norm_stats.py and scripts/train_fast_tokenizer.py, which only need
``observation.state`` / ``action`` columns, so reads are column-pruned.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

EPISODE_COLUMNS = ["observation.state", "action", "episode_index", "timestamp"]


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    path: Path
    episode_index: int | None = None  # None: the file holds exactly one episode (2025 layout)

    def __str__(self) -> str:
        return str(self.path) if self.episode_index is None else f"{self.path}#ep{self.episode_index}"


def list_v3_data_files(data_root: str | Path) -> list[Path]:
    return sorted(Path(data_root).glob("data/chunk-*/file-*.parquet"))


def v3_episode_lengths(data_root: str | Path) -> dict[int, int]:
    """Episode lengths keyed by the ``episode_index`` value stored in the data parquets.

    Read from the data files rather than ``meta/episodes`` because some exports re-index one but not
    the other (e.g. a local subset whose data column runs 0..199 while meta still carries the raw
    ids like 340020); what matters is the value a training sample carries.
    """
    lengths: dict[int, int] = {}
    for f in list_v3_data_files(data_root):
        counts = pd.read_parquet(f, columns=["episode_index"])["episode_index"].value_counts()
        for ep, n in counts.items():
            lengths[int(ep)] = lengths.get(int(ep), 0) + int(n)
    return lengths


def list_episode_sources(data_root: str | Path) -> list[EpisodeSource]:
    data_root = Path(data_root)
    legacy = sorted(data_root.glob("data/task-*/episode_*.parquet"))
    if legacy:
        return [EpisodeSource(p) for p in legacy]
    sources = []
    for f in list_v3_data_files(data_root):
        for ep in pd.read_parquet(f, columns=["episode_index"])["episode_index"].unique():
            sources.append(EpisodeSource(f, int(ep)))
    return sources


def read_episode(source: EpisodeSource | str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not isinstance(source, EpisodeSource):
        source = EpisodeSource(Path(source))
    columns = columns or EPISODE_COLUMNS
    if source.episode_index is None:
        return pd.read_parquet(source.path, columns=columns)
    df = pd.read_parquet(source.path, columns=columns, filters=[("episode_index", "==", source.episode_index)])
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True) if "timestamp" in df else df
