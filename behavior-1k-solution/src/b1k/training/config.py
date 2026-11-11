"""Training configuration for BEHAVIOR-1K challenge.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, List, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

# Import from B1K custom modules
from b1k.models import pi_behavior_config
from b1k.policies import b1k_policy
from b1k.shared import normalize as _normalize
from b1k.training import weight_loaders
from b1k import transforms as b1k_transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.
    """
    # Assets directory. If not provided, the config assets_dirs will be used.
    assets_dir: str | None = None
    # Asset id. If not provided, the repo id will be used.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    # If true, will use per-timestamp normalization for actions
    use_per_timestamp_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt (not used for PI_BEHAVIOR).
    prompt_from_task: bool = False

    # Only used for RLDS data loader.
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str | None = None

    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Episodes index to use for training 
    episodes_index: List[int] | None = None

    # Dataset layout under behavior_dataset_root:
    #   "b1k_2025"   - 2025-challenge-demos via OmniGibson's BehaviorLeRobotDataset (original pipeline)
    #   "lerobot_v3" - 2026-challenge-demos (or a subset pulled from it) via the LeRobot v3 loader
    dataset_format: Literal["b1k_2025", "lerobot_v3"] = "b1k_2025"
    # lerobot_v3 only: explicit dataset task_index -> 2026 task id overrides (auto-resolved from task names otherwise)
    task_id_map: dict[int, int] | None = None
    # lerobot_v3 only: timestamp tolerance for the LeRobot loader (B1K exports need a loose one)
    tolerance_s: float = 0.5
    # lerobot_v3 only: stage labels from equal time slices ("time", original) or from the demos' skill
    # annotations under <root>/annotations ("skills", ComputeSubtaskStateFromSkills)
    stage_labels: Literal["time", "skills"] = "time"
    # lerobot_v3 only: frames inside these skill-annotation segments (skill_description, e.g. "open lid") appear
    # oversample_factor times per epoch; () = off. Added 2026-10-12 for the task-68 lid-opening fine-tune.
    oversample_skills: tuple[str, ...] = ()
    oversample_factor: int = 1
    # lerobot_v3 only: LeRobot video backend (None = library default)
    video_backend: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for B1K."""

    default_prompt: str | None = None  # Not used (task embeddings instead)

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                b1k_transforms.ComputeSubtaskStateFromMeta(dataset=None),
                b1k_transforms.TaskIndexToTaskId(),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,  # Always use z-score normalization for B1K
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    """Data configuration for BEHAVIOR-1K dataset."""

    action_sequence_keys: Sequence[str] = ("action",)
    use_delta_joint_actions: bool = False
    
    # FAST auxiliary tokenization (only for PI_BEHAVIOR with use_fast_auxiliary)
    use_fast_tokenization: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base = self.base_config or DataConfig()
        if base.dataset_format == "lerobot_v3":
            return self._create_lerobot_v3(assets_dirs, model_config)

        # Repack transforms for B1K observations
        repack_mapping = {
            "observation/egocentric_camera": "observation.images.rgb.head",
            "observation/wrist_image_left": "observation.images.rgb.left_wrist",
            "observation/wrist_image_right": "observation.images.rgb.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "task_index": "task_index",  # Always preserve task_index
            "timestamp": "timestamp",    # Preserve timestamp for subtask state computation
            "episode_index": "episode_index",  # Preserve episode_index for episode length lookup
            "index": "index",           # Preserve index
        }
            
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        # Prepare data for policy training
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )

        # Delta action transforms
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (subtask state, task ID, padding)
        model_transforms = ModelTransformFactory()(model_config)
        
        # FAST tokenization (if enabled for PI_BEHAVIOR)
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            
            # Get base config to access norm_stats
            base_config = self.create_base_config(assets_dirs, model_config)
            
            # Only add transform if tokenizer directory exists
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(
                    f"FAST tokenizer not found at {tokenizer_path}. "
                    "FAST auxiliary training will be disabled (inference mode)."
                )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

    def _create_lerobot_v3(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Same transform stack as create(), but for a LeRobot v3 (2026-challenge-demos) root.

        Differences: tolerant key repacking (several export spellings), dataset task_index mapped to the
        2026 task id read from meta/tasks, and stage labels computed by ComputeSubtaskStateV3 (installed
        by data_loader.transform_dataset, which knows the episode lengths).
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        from b1k.training import lerobot_v3

        base = self.create_base_config(assets_dirs, model_config)
        if not base.behavior_dataset_root:
            raise ValueError("dataset_format='lerobot_v3' requires behavior_dataset_root")
        if pathlib.Path(base.behavior_dataset_root).expanduser().exists():
            meta = LeRobotDatasetMetadata(self.repo_id, root=base.behavior_dataset_root)
            task_id_map = lerobot_v3.resolve_task_id_map(lerobot_v3.tasks_from_meta(meta), base.task_id_map)
            logging.info(f"LeRobot v3 dataset task_index -> 2026 task id map: {task_id_map}")
        else:
            # Inference (serve_b1k.py on a machine without the dataset): the task id comes from the
            # eval harness observation, and TaskIndexToTaskId is dropped by create_trained_policy anyway.
            logging.warning(
                f"Dataset root {base.behavior_dataset_root} not found; assuming inference-only use "
                "(no task_index -> task id map)."
            )
            task_id_map = base.task_id_map

        repack_transform = _transforms.Group(
            inputs=[lerobot_v3.RepackFirstAvailable(lerobot_v3.REPACK_CANDIDATES_2026)]
        )
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                # Placeholder; data_loader.transform_dataset swaps in ComputeSubtaskStateV3 with episode lengths.
                b1k_transforms.ComputeSubtaskStateFromMeta(dataset=None),
                b1k_transforms.TaskIndexToTaskId(task_mapping=task_id_map),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )
        if self.use_fast_tokenization and getattr(model_config, "use_fast_auxiliary", False):
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base.norm_stats,
                        use_per_timestamp=base.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(f"FAST tokenizer not found at {tokenizer_path}; FAST auxiliary disabled.")

        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            task_id_map=task_id_map,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "B1K"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config (PI_BEHAVIOR only for B1K).
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_behavior_config.PiBehaviorConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Note: PyTorch support removed - JAX only

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotB1KDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int | None = None
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # FSDP configuration for model sharding across devices.
    fsdp_devices: int = 1
    
    # Validation configuration
    val_log_interval: int = 100
    val_batch_size: int | None = None
    val_num_batches: int = 10
    val_repo_id: str | None = None
    val_episodes_index: List[int] | None = None
    
    # Number of flow matching samples per training step
    num_flow_samples: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# B1K Training Configurations
_CONFIGS = [
    TrainConfig(
        name="pi_behavior_b1k_fast",
        exp_name="openpi",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            # FAST auxiliary training
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",  # Encode 22 dimensions
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,  # No text prompts for PI_BEHAVIOR
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=True,  # Enable per-timestamp normalization
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,  # Enable FAST tokenization in data pipeline
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=80,
        save_interval=500,
        keep_period=2000,
    ),
]

# ---------------------------------------------------------------------------------------------
# 2026 BEHAVIOR Challenge: fine-tune the 2025 winner's checkpoint on 2026-challenge-demos subsets.
#
# * num_tasks=100: task-embedding / stage tables are extended to the 2026 task list (tasks 0-49 keep
#   their rows, new tasks get rows initialised from the mean of the trained ones -- see
#   weight_loaders._extend_task_tables).
# * dataset_format="lerobot_v3": reads a LeRobot v3 root (official export or a per-task subset of it).
# * Assets: for fine-tuning, REUSE the winner's normalisation stats + FAST tokenizer so the pretrained
#   weights keep seeing the scaling they were trained with:
#     mkdir -p outputs/assets/<config name>/<repo_id>
#     cp -r <checkpoint>/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} outputs/assets/<config name>/<repo_id>/
#   (or recompute with scripts/compute_norm_stats.py --config-name <config name> --correlation).
# ---------------------------------------------------------------------------------------------
_PI_BEHAVIOR_2026_MODEL = pi_behavior_config.PiBehaviorConfig(
    action_horizon=30,
    action_dim=32,
    num_tasks=pi_behavior_config.NUM_TASKS_2026,
    use_correlated_noise=True,
    correlation_beta=0.5,
    use_fast_auxiliary=True,
    fast_loss_weight=0.05,
    fast_encoded_dims="0:6,7:23",
    fast_vocab_size=1024,
    max_fast_tokens=200,
    use_kv_transform=True,
    use_knowledge_insulation=False,
    subtask_loss_weight=0.1,
    freeze_vision_backbone=True,
)

# Winner checkpoint to fine-tune from. checkpoint_1 covers task 34; behavior_50t_checkpoint is the
# generalist trained on all 50 tasks before the split (recommended init for new tasks).
_WINNER_CKPT_LOCAL = "/media/rena/Elements/openpi_checkpoints/winner_larchenko/checkpoint_1/params"
_WINNER_CKPT_POD = "/workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/params"


def _pi_behavior_2026_config(
    name: str,
    repo_id: str,
    dataset_root: str,
    weight_params_path: str,
    *,
    episodes: List[int] | None = None,
    task_id_map: dict[int, int] | None = None,
    batch_size: int = 16,
    num_train_steps: int = 20_000,
    num_workers: int = 16,
    peak_lr: float = 5e-5,
    exp_name: str = "ft1",
    # Skill-annotation stages are the default for every 2026 run from 2026-10-11 on (user: use the new
    # training method for all future training). Runs trained before that used "time".
    stage_labels: Literal["time", "skills"] = "skills",
    oversample_skills: tuple[str, ...] = (),
    oversample_factor: int = 1,
) -> TrainConfig:
    return TrainConfig(
        name=name,
        exp_name=exp_name,
        project_name="B1K-2026",
        model=_PI_BEHAVIOR_2026_MODEL,
        data=LeRobotB1KDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root=dataset_root,
                use_per_timestamp_norm=True,
                dataset_format="lerobot_v3",
                episodes_index=episodes,
                task_id_map=task_id_map,
                tolerance_s=0.5,
                stage_labels=stage_labels,
                oversample_skills=oversample_skills,
                oversample_factor=oversample_factor,
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        ),
        # Fine-tuning: shorter, gentler schedule than the from-scratch 200k-step run.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
            decay_lr=peak_lr / 10,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(weight_params_path),
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=num_workers,
        save_interval=500,
        keep_period=2000,
    )


_CONFIGS += [
    # Local smoke test on the laptop: the hanging_pictures subset at ~/behavior-challenge-2026 (256-dim
    # 2025-layout states, observation.images.rgb.* keys). Not for real training (12 GB GPU).
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_local34",
        repo_id="b1k_2026_hanging_pictures",
        dataset_root="/home/rena/behavior-challenge-2026/datasets/hanging_pictures",
        weight_params_path=_WINNER_CKPT_LOCAL,
        batch_size=2,
        num_workers=2,
    ),
    # Pilot: task 89 installing_a_fax_machine (2-predicate goal -> partial credit measurable). Dataset
    # root is the pod's existing subset; episodes 17800-17979 == task 89 in the official export.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task89",
        repo_id="b1k_2026_task89",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task89",
        weight_params_path=_WINNER_CKPT_POD,
        episodes=list(range(17800, 17980)),
        batch_size=32,
        num_train_steps=10_000,
        num_workers=24,
    ),
    # Task 40 make_microwave_popcorn: one of the 2025 tasks (id < 50), so the winner checkpoint already
    # has a TRAINED task embedding and stage rows for it -- keep num_tasks=50 so the tables match
    # checkpoint_4 (their task-40 specialist, public-leaderboard q 0.90) exactly, with no new rows.
    # This run only has to close the 2025->2026 domain gap, so it is short and gentle.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task40",
            repo_id="b1k_2026_task40",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task40",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_4/params",
            episodes=list(range(8000, 8180)),  # 200 demos/task: task 40 -> 8000..8199; hold out the last 20
            # Same recipe as the task-89 run that reached official q=0.250: batch 16, 10k steps,
            # peak lr 5e-5 with 500 warmup and cosine decay over the run, 15 flow samples, EMA 0.99.
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 46 cook_bacon: also a 2025 task (id < 50), covered by the winner's checkpoint_3 specialist
    # (tasks 4, 27, 31-39, 41, 46, 49; public-leaderboard q 0.76 on this task). Same recipe as task 89/40.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task46",
            repo_id="b1k_2026_task46",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task46",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_3/params",
            episodes=list(range(9200, 9380)),  # task 46 -> episodes 9200..9399; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 10 set_up_a_coffee_station_in_your_kitchen: 2025 task, winner's checkpoint_1 specialist
    # (tasks 2,3,5,6,10,11,13,14,15,19,23,24,25,28,29,34,42,44,47,48). Chosen on our own eval evidence:
    # 6 goal predicates (nextto/ontop placements -> partial credit, the task-56 profile that scored
    # 0.263) and the shortest human-demo horizon (6266 steps) of any untried multi-predicate task.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task10",
            repo_id="b1k_2026_task10",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task10",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params",
            episodes=list(range(2000, 2180)),  # task 10 -> episodes 2000..2199; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 42 chop_an_onion: 2025 task, winner's checkpoint_1 (already on the pod from task 10).
    # Picked with the same two predictors: 6400-step human-demo horizon (shortest untried
    # multi-predicate task after task 10) and 4 goal predicates, of which two -- inside(parer, sink)
    # and inside(chopping_board, sink) -- are plain placements that earn partial credit even if the
    # hard dicing step (real / contains) never lands. That is the task-56 profile.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task42",
            repo_id="b1k_2026_task42",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task42",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params",
            episodes=list(range(8400, 8580)),  # task 42 -> episodes 8400..8599; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 0 turning_on_radio: winner's checkpoint_2 (tasks 0,1,7,8,9,12,16,17,18,20,21,22,26,30,43,45).
    # Shortest horizon in the entire 100-task set: 2150 human-demo steps, two thirds shorter than
    # task 40 (3238) which is our best score at 0.350. Single goal predicate toggled_on(radio), and
    # task 89 proves that predicate is reachable (it needed ontop + toggled_on and solved 4 of 20
    # outright). Crucially the scene holds ONE radio and no look-alike appliance, so it cannot hit
    # the grounding failure that sank task 46, where the robot opened the oven instead of the stove
    # -- this architecture has no language conditioning to disambiguate similar objects.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task0",
            repo_id="b1k_2026_task0",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task0",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params",
            episodes=list(range(0, 180)),  # task 0 -> episodes 0..199; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 1 picking_up_trash: winner's checkpoint_2 (tasks 0,1,7,8,9,12,16,17,18,20,21,22,26,30,43,45).
    # 5268-step horizon — the shortest untried task with GRADED credit, and close to the <4600 band
    # where every score we have came from. Goal is a forall over three soda cans, each of which must
    # end up inside the ashcan, so it scores in thirds the way task 56's four predicates scored in
    # quarters (0.263 from partials alone) rather than all-or-nothing.
    # Three pick-and-places is also the skill we are demonstrably best at (task 40: 7 outright
    # solves), and a soda can vs an ashcan cannot be confused the way the oven was for the stove.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task1",
            repo_id="b1k_2026_task1",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task1",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params",
            episodes=list(range(200, 380)),  # task 1 -> episodes 200..399; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 22 putting_shoes_on_rack (2026-09-28). Same recipe as task 1, our best task (0.517 = 78% of
    # the 2025 winner's 0.667 on it): init from winner checkpoint_2, which already trained task 22, and
    # num_tasks=50 so its trained task-22 embedding is kept. The winner scored 0.61 on task 22
    # (public 0.50 / hidden 0.72), the highest of any untried partial-credit task. Goal: 4 shoes
    # touching the hallstand and off the floor (2 predicates per shoe, all false at reset) + 2 pairs
    # side by side = 10 predicates, so each shoe racked is worth 0.2 on its own. Demo 7726 steps.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task22",
            repo_id="b1k_2026_task22",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task22",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params",
            episodes=list(range(4400, 4580)),  # task 22 -> episodes 4400..4599; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 90 composting_waste (2026-09-28). Picked from 50-99 on the profile of our scoring new tasks
    # (89: 4077 demo steps, 56: 4532): 3639-step demos, and two plain placements (half banana and
    # half pomegranate inside the ashcan, 0.5 each, both false at reset) -- the same put-it-in-the-bin
    # skill as task 1, our best task. Recipe = task 89's: 50-task generalist init (no specialist exists
    # for ids >= 50), 10k steps, batch 16, lr 5e-5.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task90",
        repo_id="b1k_2026_task90",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task90",
        weight_params_path=_WINNER_CKPT_POD,          # 50-task generalist; no specialist exists for id >= 50
        episodes=list(range(18000, 18180)),           # task 90 -> episodes 18000..18199; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 67 boxing_food_after_dinner (2026-09-28). Picked from the team tracker's UNCLAIMED tasks >= 50
    # (90/77/22/59 were taken). 6698-step demos; 6 goal predicates, 5 earnable (fridge-closed is true at
    # reset): 2 plates countertop -> sink (independent, easy), 2 tacos -> tupperware, tupperware
    # (starts in a cabinet) -> fridge. Recipe = task 89's: generalist init, 10k steps, batch 16.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task67",
        repo_id="b1k_2026_task67",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task67",
        weight_params_path=_WINNER_CKPT_POD,          # 50-task generalist; no specialist exists for id >= 50
        episodes=list(range(13400, 13580)),           # task 67 -> episodes 13400..13599; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 12 preparing_lunch_box (2026-09-29). Task-1 recipe (our best task): init winner checkpoint_2,
    # which already trained task 12 (winner 2025: 0.517 public / 0.367 hidden), num_tasks=50 so its
    # task-12 embedding is kept. Goal: 6 predicates, 5 earnable (fridge-closed true at reset); 4 of them
    # are items on a chopping board next to the packing box (2 apple halves, sandwich, cookie -> inside
    # box) = task 1's put-it-in-the-container skill with little navigation; tea bottle is in the fridge.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task12",
            repo_id="b1k_2026_task12",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task12",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params",
            episodes=list(range(2400, 2580)),  # task 12 -> episodes 2400..2599; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 20 sorting_vegetables (2026-09-30, Rena). Task-1 recipe: init winner checkpoint_2, which already
    # trained task 20 (winner 2025: 0.48 public / 0.45 hidden), num_tasks=50 so its task-20 embedding is
    # kept. Goal: 13 vegetables from two wicker baskets on the floor into 3 mixing bowls by type = 13 inside
    # predicates, all false at reset, 1/13 each -- task 1's put-it-in-a-container skill, repeated.
    # checkpoint_2 is the family that transferred best for us (task 1 0.567, Richard's task 12 screening
    # 0.20/instance). Long demos (11,904 steps) -> ~2 h eval rollouts.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task20",
            repo_id="b1k_2026_task20",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task20",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params",
            episodes=list(range(4000, 4180)),  # task 20 -> episodes 4000..4199; hold out the last 20
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 99 sorting_books_on_shelf, RETRAIN as a dedicated single-task run (2026-09-26).
    # It previously shared a 20k-step run with tasks 56 and 78, so its task embedding got roughly a
    # third of the gradient signal; task 56 came out of that run at 0.263 while 99 scored 0.000.
    # Task 89 — one task, one run — produced four outright solves from the same generalist init,
    # which is the hypothesis being tested here.
    # Worth keeping in view: 99 has the most generous payout structure we have seen (10 graded
    # placements, 0.1 per book) but the longest horizon of any task we have scored on (7739 demo
    # steps vs the <4600 band). The eval reaches stage 4 of 6 and never lands a book, so the target
    # is one successful placement, not completion.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task99",
        repo_id="b1k_2026_task99",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task99",
        weight_params_path=_WINNER_CKPT_POD,          # 50-task generalist; no specialist exists for id >= 50
        episodes=list(range(19800, 19980)),           # task 99 -> episodes 19800..19999; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 59 organizing_art_supplies (2026-09-26). Picked from 51-88 by the partial-credit rule:
    # 5 ground goal predicates (4 items inside the carryall + carryall on the desk), ALL false at
    # reset, each a plain pick-and-place worth 0.2 on its own. The four items start together on the
    # desk, so little navigation. 6194-step demo horizon, same class as task 10 (6266, scored 0.15).
    # Recipe = task 89's as actually launched (generalist init, 10k steps, batch 16, lr 5e-5): the
    # only run of ours that took a task id >= 50 from zero to outright solves.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task59",
        repo_id="b1k_2026_task59",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task59",
        weight_params_path=_WINNER_CKPT_POD,          # 50-task generalist; no specialist exists for id >= 50
        episodes=list(range(11800, 11980)),           # task 59 -> episodes 11800..11999; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 62 (halve_an_egg), claimed 2026-09-28. 5 goal predicates, all false at reset:
    # 2 ontop + 1 inside are plain pick-and-place and pay independently (0.6 reachable without
    # ever cutting); the 2 'real' predicates need the halve. Demo 6,377 steps, just inside the
    # ~6.5k limit above which nothing has scored. Generalist init: no specialist exists for id >= 50,
    # and the weight loader pads the task tables 50->100 so task 62 gets its own embedding row.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task62",
        repo_id="b1k_2026_task62",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task62",
        weight_params_path=_WINNER_CKPT_POD,
        episodes=list(range(12400, 12580)),           # task 62 -> episodes 12400..12599; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Batch requested by the user 2026-10-08: tasks 26, 41, 43, 49, 51. Same recipe as every other
    # single-task run (10k steps, batch 16, lr 5e-5, ft1). Ids < 50 start from the winner specialist
    # that already trained them and keep num_tasks=50; 51 starts from the 50-task generalist.
    *[
        dataclasses.replace(
            _pi_behavior_2026_config(
                f"pi_behavior_b1k_2026_task{task}",
                repo_id=f"b1k_2026_task{task}",
                dataset_root=f"/workspace/behavior-challenge-2026/datasets/task{task}",
                weight_params_path=f"/workspace/openpi_checkpoints/winner_larchenko/{ckpt}/params",
                episodes=list(range(task * 200, task * 200 + 180)),  # hold out the last 20 demos
                batch_size=batch,
                num_train_steps=steps,
                num_workers=16,
                peak_lr=5e-5,
            ),
            model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        )
        # 26 assembling_gift_baskets (16 inside goals, winner 0.21/0.28), 43 slicing_vegetables
        # (winner 0.19/0.28): checkpoint_2. 41 cook_cabbage, 49 make_pizza (winner 0.0 on both):
        # checkpoint_3. Tasks 41 and 49 run 20k steps at batch 16 (user, 2026-10-10); the cosine decay follows
        # num_train_steps.
        for task, ckpt, steps, batch in (
            (26, "checkpoint_2", 10_000, 16),
            (43, "checkpoint_2", 10_000, 16),
            (41, "checkpoint_3", 20_000, 16),
            (49, "checkpoint_3", 20_000, 16),
            # 34 hanging_pictures retrain with skill stages (user, 2026-10-11): the scored 0.0 came from
            # the old pi05 3task_calib run; this is the first run with the winner-specialist recipe.
            (34, "checkpoint_1", 20_000, 16),
        )
    ],
    # 51 cook_a_brisket: id >= 50, so no specialist exists -> task-89 recipe with the generalist init.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task51",
        repo_id="b1k_2026_task51",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task51",
        weight_params_path=_WINNER_CKPT_POD,
        episodes=list(range(10200, 10380)),           # task 51 -> episodes 10200..10399; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # 68 cleaning_up_branches_and_twigs (Richard's task). Reconstructed 2026-10-11 from his checkpoints after his pod's
    # container (and his config copy) was wiped: task_embeddings [100, 2048] / task_stage_embeddings [1105, 1024] =
    # the 100-task 2026 model with the 50t generalist init, i.e. the task-51 recipe; norm stats ship in the
    # checkpoint (assets/b1k_2026_task68). Used for eval; training details beyond that are a best guess.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task68",
        repo_id="b1k_2026_task68",
        # B1K_DATASETS_ROOT: per-pod dataset location (task 68 retrain keeps it in /dev/shm on an S3-volume pod)
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task68",
        # B1K_INIT_PARAMS: local copy of the init params (S3/geesefs volume reads failed with RESOURCE_EXHAUSTED)
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", _WINNER_CKPT_POD),
        episodes=list(range(68 * 200, 68 * 200 + 180)),
        batch_size=16,
        # Retrain 2026-10-12 (user): all four of Richard's checkpoints scored 0 on 311-313 -> winner 50t init,
        # 20k steps, skill-annotation stages (the default).
        num_train_steps=20_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 54 putting_away_toys (Richard's task; trained on his pod 2026-10-09, 8 inside goals, 11,297-step demo).
    # Same recipe as task 64 at the time: 50t generalist init (the task tables are padded 50 -> 100), 10k steps,
    # batch 16, peak lr 5e-5. Bootstrap: pipeline/bootstrap_task54_training.sh (exp ft1, ckpts 2000-9999).
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task54",
        repo_id="b1k_2026_task54",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task54",
        weight_params_path=_WINNER_CKPT_POD,
        episodes=list(range(10800, 10980)),           # task 54 -> episodes 10800..10999; hold out the last 20
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 51 again with stage labels from the demos' skill annotations instead of time slices (user,
    # 2026-10-11: the time-sliced run scored 0 and its eval stage kept sliding back). Same recipe and
    # init otherwise, so the two runs compare directly.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task51_skills",
        repo_id="b1k_2026_task51",
        dataset_root="/workspace/behavior-challenge-2026/datasets/task51",
        weight_params_path=_WINNER_CKPT_POD,
        episodes=list(range(10200, 10380)),
        batch_size=16,
        num_train_steps=10_000,
        num_workers=16,
        peak_lr=5e-5,
        stage_labels="skills",
    ),
    # Task 43 retrain with skill stages, 20k (user, 2026-10-11): the time-sliced 10k run scored 0 on 311 and
    # its video shows it opening the fridge, never picking a vegetable, and wandering off. Same init
    # (winner specialist checkpoint_2, num_tasks=50) and episodes as pi_behavior_b1k_2026_task43; keep
    # 10000 to separate the effect of skill stages (vs old 9999) from that of more steps (vs 19999).
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task43_skills",
            repo_id="b1k_2026_task43",
            dataset_root="/workspace/behavior-challenge-2026/datasets/task43",
            weight_params_path="/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params",
            episodes=list(range(43 * 200, 43 * 200 + 180)),
            batch_size=16,
            num_train_steps=20_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # 61 bringing_paper_to_recycling: retrain with skill stages, 20k (user 2026-10-11; Richard's first run used time
    # stages). Same recipe as task 68: 100-task model, winner 50t generalist init.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task61",
        repo_id="b1k_2026_task61",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task61",
        # B1K_INIT_PARAMS: local copy of the init params (S3/geesefs volume reads failed with RESOURCE_EXHAUSTED)
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", _WINNER_CKPT_POD),
        episodes=list(range(61 * 200, 61 * 200 + 180)),
        batch_size=16,
        num_train_steps=20_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # 64 setting_the_table: first training (user 2026-10-12 "retrain" for task 64). Same recipe as 61/68: 100-task model,
    # winner 50t generalist init, skill stages (default), 20k steps.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task64",
        repo_id="b1k_2026_task64",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task64",
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", _WINNER_CKPT_POD),
        episodes=list(range(64 * 200, 64 * 200 + 180)),
        batch_size=16,
        num_train_steps=20_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # 69 vacuuming_floors (user 2026-10-15 ~11:55 PDT: "Train 69 with our new way. 30k steps, use the last checkpoint do eval
    # directly"): new 2026 task (no winner specialist) -> winner 50t generalist init like 64, but the single-task LOW-lr recipe
    # (peak 5e-6 -> 1e-6 cosine, warmup 100), skill stages, 30k steps, final 29999 straight to the official run. Goal = 1 literal.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task69_lowlr",
            repo_id="b1k_2026_task69_lowlr",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task69_lowlr",
            weight_params_path=os.environ.get("B1K_INIT_PARAMS", _WINNER_CKPT_POD),
            episodes=list(range(69 * 200, 69 * 200 + 180)),
            batch_size=16,
            num_train_steps=30_000,
            num_workers=16,
            peak_lr=5e-6,
            stage_labels="skills",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=30_000, decay_lr=1e-6),
    ),
    # 92 installing_a_scanner (user 2026-10-16 ~00:30 PDT: "train 92 with 89 fine tune, 30k step"; lr: user chose 5e-5 over
    # 2.5e-5/5e-6 ~00:40): init = OUR task-89 model (installing_a_fax_machine, ft_task89/8000, official 0.250; 100-task tables,
    # same 5-stage layout as 92), not the winner. Same lr recipe as 89 (peak 5e-5, warmup 500, cosine -> 5e-6), skill stages,
    # 30k steps; B1K_INIT_PARAMS points at the downloaded Rena-Tian/b1k-2026-task89-ft1 params.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task92",
        repo_id="b1k_2026_task92",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task92",
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/ft_task89/8000/params"),
        episodes=list(range(92 * 200, 92 * 200 + 180)),
        batch_size=16,
        num_train_steps=30_000,
        num_workers=16,
        peak_lr=5e-5,
        stage_labels="skills",
    ),
    # 92 continuation (user 2026-10-16 ~10:25 PDT: "after finishing 30k, then keep training to 50k"): starts from the 30k run's
    # final params (step 29999), 20k more steps (global 30000 -> 50000). Short warmup to 1e-5 (2x where the 30k cosine ended),
    # cosine -> 2e-6. Params-only ckpts, so the optimizer state starts fresh.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task92_cont",
            repo_id="b1k_2026_task92_cont",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task92_cont",
            weight_params_path=os.environ.get("B1K_INIT_PARAMS", "/dev/shm/winner/task92_29999/params"),
            episodes=list(range(92 * 200, 92 * 200 + 180)),
            batch_size=16,
            num_train_steps=20_000,
            num_workers=16,
            peak_lr=1e-5,
            stage_labels="skills",
            # user 2026-10-16 ~14:25 PDT: screens show the robot carrying the scanner upright and never setting it down to
            # press the button -> repeat the "turn on switch" + "place on" demo frames 4x (scanner flat on the desk, press).
            oversample_skills=("turn on switch", "place on"),
            oversample_factor=4,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=1e-5, decay_steps=20_000, decay_lr=2e-6),
    ),
    # 65 (Amy's task; user 2026-10-13: new A100 pod ur41wgi01rxhoq "should be for 65 training"). Same recipe as 61/64:
    # 100-task model, winner 50t generalist init, skill stages (default), 20k steps, episodes 13000-13179.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task65",
        repo_id="b1k_2026_task65",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task65",
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", _WINNER_CKPT_POD),
        episodes=list(range(65 * 200, 65 * 200 + 180)),
        batch_size=16,
        num_train_steps=20_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # Task 45 cook_hot_dogs (Thomas's task; user 2026-10-13: "train it with 2026 demo 20k, using our runbook"). The winner's
    # checkpoint_2 unchanged scored 0 on 301/302 (never takes the hot dogs out of the fridge) -> runbook recipe: specialist
    # init checkpoint_2, num_tasks 50, 20k steps, batch 16, lr 5e-5, skill stages, episodes 9000-9179.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task45",
            repo_id="b1k_2026_task45",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task45",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=list(range(45 * 200, 45 * 200 + 180)),
            batch_size=16,
            num_train_steps=20_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Tasks 7 + 16 + 18 + 30 + 43 trained together, 10k steps (user 2026-10-14 ~08:30 PDT, replaces the 500-step plan; the
    # coach ruled last year's checkpoints may not be used unchanged). All five are checkpoint_2 tasks -> same recipe as 8+9.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task7_16_18_30_43",
            repo_id="b1k_2026_task7_16_18_30_43",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task7_16_18_30_43",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=[e for t in (7, 16, 18, 30, 43) for e in range(t * 200, t * 200 + 180)],
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Tasks 11 + 13 + 15 + 17 + 25 (Thomas's 0-point tasks, now Rena's) trained together, 10k steps (user 2026-10-14 ~10:05 PDT,
    # pod mti). 11/13/15/25 are winner checkpoint_1 tasks (17 is checkpoint_2) -> init checkpoint_1; same recipe as 7+16+18+30+43.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task11_13_15_17_25",
            repo_id="b1k_2026_task11_13_15_17_25",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task11_13_15_17_25",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params"),
            episodes=[e for t in (11, 13, 15, 17, 25) for e in range(t * 200, t * 200 + 180)],
            batch_size=16,
            num_train_steps=10_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 16 alone, LOW-lr fine-tune from winner checkpoint_2 (user 2026-10-14 ~17:10 PDT). The 5-task 10k run (peak 5e-5) dropped
    # task 16 from 0.600 (checkpoint_2 unchanged) to 0.250: it kept carrying boxes but lost the final stacking. 3000 steps, peak
    # 5e-6 -> 1e-6 cosine, warmup 100, skill stages; on mti (A100) B1K_INIT_PARAMS points at a local copy of checkpoint_2.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task16_lowlr",
            repo_id="b1k_2026_task16_lowlr",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task16_lowlr",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=list(range(16 * 200, 16 * 200 + 180)),
            batch_size=16,
            num_train_steps=3_000,
            num_workers=16,
            peak_lr=5e-6,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=3_000, decay_lr=1e-6),
    ),
    # Task 7 alone, same LOW-lr recipe as task16_lowlr (user 2026-10-14 ~17:20 PDT: "train the same way for task 7"). checkpoint_2
    # unchanged scored 0.4167 on 301-310; the 5-task 10k run scored 0 on every screen.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task7_lowlr",
            repo_id="b1k_2026_task7_lowlr",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task7_lowlr",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=list(range(7 * 200, 7 * 200 + 180)),
            batch_size=16,
            num_train_steps=3_000,
            num_workers=16,
            peak_lr=5e-6,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=3_000, decay_lr=1e-6),
    ),
    # Task 11 alone, same LOW-lr recipe as task16_lowlr but from winner checkpoint_1 (task 11 is a checkpoint_1 task)
    # (user 2026-10-14 ~20:55 PDT: "train 11"). The 11/13/15/17/25 group run (5e-5, 10k) scored 0 on its step-5000 screens.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task11_lowlr",
            repo_id="b1k_2026_task11_lowlr",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task11_lowlr",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params"),
            episodes=list(range(11 * 200, 11 * 200 + 180)),
            batch_size=16,
            num_train_steps=3_000,
            num_workers=16,
            peak_lr=5e-6,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=3_000, decay_lr=1e-6),
    ),
    # Task 17 alone, same LOW-lr recipe as task16_lowlr but 2000 steps (user 2026-10-14 ~22:40 PDT: "train task 17", "task 17 2000step"; task16_lowlr scored 0.500).
    # checkpoint_2 task; winner 2025: 0.267 public / 0.667 hidden. The 11/13/15/17/25 group run scored 0 on its step-5000 screens.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task17_lowlr",
            repo_id="b1k_2026_task17_lowlr",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task17_lowlr",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=list(range(17 * 200, 17 * 200 + 180)),
            batch_size=16,
            num_train_steps=2_000,
            num_workers=16,
            peak_lr=5e-6,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=2_000, decay_lr=1e-6),
    ),
    # Task 13 alone, same LOW-lr 2000-step recipe as task17_lowlr but from winner checkpoint_1 (task 13 is a checkpoint_1 task)
    # (user 2026-10-14 ~23:00 PDT: "que 11,18,30,17,13"). Winner 2025: 0.200 public / 0.167 hidden; group run: 0 at step 5000.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task13_lowlr",
            repo_id="b1k_2026_task13_lowlr",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task13_lowlr",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params"),
            episodes=list(range(13 * 200, 13 * 200 + 180)),
            batch_size=16,
            num_train_steps=2_000,
            num_workers=16,
            peak_lr=5e-6,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=2_000, decay_lr=1e-6),
    ),
    # Tasks 15 and 25 alone, same LOW-lr 2000-step recipe as task13_lowlr (checkpoint_1 tasks). Trained only if the
    # 11/13/15/17/25 group's step-9999 screens all score 0 for that task (user 2026-10-14 ~23:05 PDT).
    *[
        dataclasses.replace(
            _pi_behavior_2026_config(
                f"pi_behavior_b1k_2026_task{_t}_lowlr",
                repo_id=f"b1k_2026_task{_t}_lowlr",
                dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + f"/task{_t}_lowlr",
                weight_params_path=os.environ.get(
                    "B1K_INIT_PARAMS", f"/workspace/openpi_checkpoints/winner_larchenko/checkpoint_{3 if _t in (4, 36, 37, 46) else 2 if _t in (21,) else 1}/params"),
                episodes=list(range(_t * 200, _t * 200 + 180)),
                batch_size=16,
                num_train_steps=2_000,
                num_workers=16,
                peak_lr=5e-6,
                stage_labels="skills",
            ),
            model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=2_000, decay_lr=1e-6),
        )
        # 29 clean_up_your_desk (user 2026-10-15 ~06:30 PDT: "start task 29 training"); 23 boxing_books_up_for_storage on idle mti
        # (~07:15); 48 canning_food (checkpoint_1) and 4 can_meat / 36 clean_a_patio / 37 clean_a_trumpet (checkpoint_3), the
        # unassigned tasks (user ~08:15: "can we try more tasks?")
        # 21 collecting_childrens_toys (checkpoint_2; user ~09:00 "try task 21 right now": Thomas's generalist got .0143, winner .433)
        # 24 storing_food + 14 carrying_in_groceries (checkpoint_1; user ~09:05 "yes": Thomas's generalist .0125 / .05, winner .375 / .15)
        # 46 cook_bacon (checkpoint_3; user ~11:50 "Run 46"): winner .757 pub; our Sept 10k lr-5e-5 run scored 0
        for _t in (15, 25, 29, 23, 48, 4, 36, 37, 21, 24, 14, 46)
    ],
    # Tasks 18 and 30 alone, same LOW-lr recipe + hard skill segments repeated 3x (user 2026-10-14 ~19:10 PDT, after the video
    # review: rollouts stall re-grasping the book / newspaper, sandals never released at the bed, lighter rarely ignited).
    # checkpoint_2 unchanged: task 18 0.367 on 301-310; task 30 0/0.125/0.5/0.5/0.25 on 309-313. 5-task 10k run: 0 on both.
    *[
        dataclasses.replace(
            _pi_behavior_2026_config(
                f"pi_behavior_b1k_2026_task{_t}_lowlr",
                repo_id=f"b1k_2026_task{_t}_lowlr",
                dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + f"/task{_t}_lowlr",
                weight_params_path=os.environ.get(
                    "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
                episodes=list(range(_t * 200, _t * 200 + 180)),
                batch_size=16,
                num_train_steps=3_000,
                num_workers=16,
                peak_lr=5e-6,
                stage_labels="skills",
                oversample_skills=_skills,
                oversample_factor=3,
            ),
            model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=3_000, decay_lr=1e-6),
        )
        for _t, _skills in (
            (18, ("pick up from", "place on next to", "place on")),
            (30, ("pick up from", "turn on switch", "ignite", "turn off switch")),
        )
    ],
    # Short fine-tunes (user 2026-10-14 ~08:20 PDT, after the coach ruled that last year's checkpoints may not be used
    # UNCHANGED): tasks 16, 7, 18, 30, 43, each from winner checkpoint_2, 500 steps, runbook data/batch/lr, warmup 100
    # (500 would leave no cosine phase). One task per run; final step 499 goes straight to the official 301-310 run.
    *[
        dataclasses.replace(
            _pi_behavior_2026_config(
                f"pi_behavior_b1k_2026_task{_t}_s500",
                repo_id=f"b1k_2026_task{_t}_s500",
                dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + f"/task{_t}_s500",
                weight_params_path=os.environ.get(
                    "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
                episodes=list(range(_t * 200, _t * 200 + 180)),
                batch_size=16,
                num_train_steps=500,
                num_workers=16,
                peak_lr=5e-5,
                stage_labels="skills",
            ),
            model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-5, decay_steps=500, decay_lr=5e-6),
        )
        for _t in (16, 7, 18, 30, 43)
    ],
    # Task 45 warm restart (2026-10-14 ~02:55 PDT): the 20k run on pod mti deadlocked inside the orbax save at step 13500
    # (no I/O, GPU idle). Checkpoints are params-only, so it cannot be --resumed: continue from ft1/13000 params
    # (B1K_INIT_PARAMS) for the remaining 7000 steps, picking the cosine up where it was (lr(13000) = 1.784e-5) and decaying
    # to the original end lr 5e-6; short warmup to rebuild the AdamW moments. Same data/stages. Local step 3000 = global
    # 16000, 6999 = global 19999.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task45_r13k",
            repo_id="b1k_2026_task45",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task45",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/root/init45/13000/params"),
            episodes=list(range(45 * 200, 45 * 200 + 180)),
            batch_size=16,
            num_train_steps=7_000,
            num_workers=16,
            peak_lr=1.784e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1.784e-5, decay_steps=7_000, decay_lr=5e-6),
    ),
    # Task 45 second warm restart (2026-10-14 ~04:00 PDT): r13k deadlocked the same way at its step-1000 save. Continue from
    # r13k/500 (= global 13500) for 6500 steps on the same curve (lr_r13k(500) = 1.773e-5 -> 5e-6), launched with
    # --save_interval=2500 so it only saves at local 2500 (= global 16000), 5000 and the final 6499 (= global 19999).
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task45_r13k5",
            repo_id="b1k_2026_task45",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task45",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/root/init45/r13k500/params"),
            episodes=list(range(45 * 200, 45 * 200 + 180)),
            batch_size=16,
            num_train_steps=6_500,
            num_workers=16,
            peak_lr=1.773e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50, peak_lr=1.773e-5, decay_steps=6_500, decay_lr=5e-6),
    ),
    # 64 continued +30k (user 2026-10-13: "set up the task-64 continuation on e0qa"): eval videos of ft_task64/19999 show the
    # policy never picking anything up (undertrained: 20k x b16 < 0.1 epoch of 3.2M frames). Starts from ft_task64/19999
    # (B1K_INIT_PARAMS, local copy on the pod), fresh warmup + cosine over 30k, skill stages, same data.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task64_cont",
        repo_id="b1k_2026_task64_cont",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task64_cont",
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", "/root/init64/19999/params"),
        episodes=list(range(64 * 200, 64 * 200 + 180)),
        batch_size=16,
        num_train_steps=30_000,
        num_workers=16,
        peak_lr=5e-5,
    ),
    # 68 lid fine-tune (user 2026-10-12 "add task 68 fine tuning training at last priority"): eval videos showed the
    # policy tipping the wheelie bin while opening its lid, so "open lid" and "place in" frames are oversampled 4x.
    # Starts from ft_task68/19999 (B1K_INIT_PARAMS overrides the path), 8k steps at half the usual peak lr.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task68_lidft",
        repo_id="b1k_2026_task68",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task68",
        weight_params_path=os.environ.get("B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/ft_task68_19999/params"),
        episodes=list(range(68 * 200, 68 * 200 + 180)),
        batch_size=16,
        num_train_steps=8_000,
        num_workers=16,
        peak_lr=2.5e-5,
        oversample_skills=("open lid", "place in"),
        oversample_factor=4,
    ),
    # Task 51 skills, continued (user 2026-10-11: "if all three 0, please continue train 51 15k steps more"): starts from
    # ft_task51_skills/9999 (uploaded to the pod at this path) and trains 15k more with the same recipe and skill
    # stages. Fresh warmup + cosine over the 15k (params-only checkpoints carry no optimizer state).
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_task51_skills_cont",
        repo_id="b1k_2026_task51",
        dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task51",
        weight_params_path="/workspace/openpi_checkpoints/ft_task51_skills_9999/params",
        episodes=list(range(10200, 10380)),
        batch_size=16,
        num_train_steps=15_000,
        num_workers=16,
        peak_lr=5e-5,
        stage_labels="skills",
    ),
    # Tasks 8 rearranging_kitchen_furniture + 9 putting_up_Christmas_decorations_inside trained TOGETHER (user
    # 2026-10-13: "train task 8,9 together with 50k steps"). Both are 2025 tasks covered by the winner's checkpoint_2
    # specialist -> init from it, keep num_tasks=50. One dataset root holds chunk-008 + chunk-009; each sample's
    # task_index (8 or 9) picks its task embedding. Skill annotations task-0008 + task-0009 pair by sorted order
    # (200 + 200 files for episodes 1600-1999).
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task8_9",
            repo_id="b1k_2026_task8_9",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task8_9",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=list(range(8 * 200, 8 * 200 + 180)) + list(range(9 * 200, 9 * 200 + 180)),
            batch_size=16,
            num_train_steps=50_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Tasks 7 picking_up_toys + 18 tidying_bedroom + 30 setting_the_fire trained together, 50k steps (user 2026-10-13,
    # if the unchanged winner checkpoint_2 gate on 311-313 is not good). All three are checkpoint_2 tasks -> same recipe as 8+9.
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task7_18_30",
            repo_id="b1k_2026_task7_18_30",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task7_18_30",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params"),
            episodes=[e for t in (7, 18, 30) for e in range(t * 200, t * 200 + 180)],
            batch_size=16,
            num_train_steps=50_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # Task 32 wash_a_baseball_cap, 50k steps (user 2026-10-13). 2025 task covered by the winner's checkpoint_3 specialist
    # (winner q 0.45 public / 0.35 hidden, 30% full successes) -> init from it, keep num_tasks=50. B1K_INIT_PARAMS: local
    # copy on geesefs pods (volume reads failed with RESOURCE_EXHAUSTED, task 68).
    dataclasses.replace(
        _pi_behavior_2026_config(
            "pi_behavior_b1k_2026_task32",
            repo_id="b1k_2026_task32",
            dataset_root=os.environ.get("B1K_DATASETS_ROOT", "/workspace/behavior-challenge-2026/datasets") + "/task32",
            weight_params_path=os.environ.get(
                "B1K_INIT_PARAMS", "/workspace/openpi_checkpoints/winner_larchenko/checkpoint_3/params"),
            episodes=list(range(32 * 200, 32 * 200 + 180)),
            batch_size=16,
            num_train_steps=50_000,
            num_workers=16,
            peak_lr=5e-5,
            stage_labels="skills",
        ),
        model=dataclasses.replace(_PI_BEHAVIOR_2026_MODEL, num_tasks=pi_behavior_config.NUM_TASKS_2025),
    ),
    # 3more_tasks subset: 56 make_rose_centerpieces, 78 make_cabinet_doors, 99 sorting_books_on_shelf.
    _pi_behavior_2026_config(
        "pi_behavior_b1k_2026_3more",
        repo_id="b1k_2026_3more_tasks",
        dataset_root="/workspace/behavior-challenge-2026/datasets/3more_tasks",
        weight_params_path=_WINNER_CKPT_POD,
        batch_size=32,
        num_train_steps=20_000,
        num_workers=24,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

