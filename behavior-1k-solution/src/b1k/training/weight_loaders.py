"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


# Row tables that grow when a checkpoint is loaded into a model with more tasks. New rows are
# appended (the 2026 task list keeps the 2025 tasks at indices 0-49 and stage_layout() appends the
# new tasks' stage rows after the existing ones), so the loaded rows keep their indices.
_EXTENDABLE_TABLES = ("task_embeddings/embedding", "task_stage_embeddings/embedding")


def _extend_task_tables(loaded_params: at.Params, params: at.Params, *, seed: int) -> at.Params:
    """Pad task-indexed embedding tables in `loaded_params` up to the reference row count.

    New rows are initialised to the mean of the loaded rows plus small Gaussian noise (10% of the
    per-feature std), i.e. a "generic task" prior rather than the random init a fresh model gets.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    rng = np.random.default_rng(seed)
    for k, v in list(flat_loaded.items()):
        if not any(k.endswith(t) for t in _EXTENDABLE_TABLES) or k not in flat_ref:
            continue
        ref_shape = flat_ref[k].shape
        v = np.asarray(v)
        if v.shape == tuple(ref_shape):
            continue
        if v.ndim != len(ref_shape) or v.shape[1:] != tuple(ref_shape[1:]) or v.shape[0] > ref_shape[0]:
            raise ValueError(f"Cannot extend {k}: checkpoint shape {v.shape} vs model shape {tuple(ref_shape)}")
        n_new = ref_shape[0] - v.shape[0]
        v32 = v.astype(np.float32)
        mean, std = v32.mean(axis=0, keepdims=True), v32.std(axis=0, keepdims=True)
        new_rows = mean + 0.1 * std * rng.standard_normal((n_new, v.shape[1])).astype(np.float32)
        flat_loaded[k] = np.concatenate([v, new_rows.astype(v.dtype)], axis=0)
        logger.info(f"Extended {k}: {v.shape[0]} -> {ref_shape[0]} rows ({n_new} new rows init from mean of loaded rows)")
    return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str
    # Seed for initialising rows of task tables that are missing from the checkpoint (new tasks).
    new_task_init_seed: int = 0

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        params_path = download.maybe_download(self.params_path)
        
        # Restore as host numpy arrays. openpi's restore_params passes explicit restore args, which a
        # bare PyTreeCheckpointer().restore() lacks -- that fails on checkpoints saved from a multi-GPU
        # (FSDP) run when restoring on a different device set (e.g. the winner's 8xH200 checkpoints).
        loaded_params = _model.restore_params(params_path, restore_type=np.ndarray)
        
        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # Loading PI_BEHAVIOR checkpoint - load ALL weights from checkpoint
            logging.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
            # A checkpoint trained with fewer tasks (e.g. the 50-task 2025 winner) has shorter
            # task_embeddings / task_stage_embeddings tables than a 100-task model; extend them.
            loaded_params = _extend_task_tables(loaded_params, params, seed=self.new_task_init_seed)
            # Use _merge_params with empty missing_regex to validate shapes
            return _merge_params(loaded_params, params, missing_regex="^$")
        else:
            # Loading Pi05 checkpoint - preserve new PI_BEHAVIOR-specific parameters
            logging.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")
            
            # These parameters are NEW in PI_BEHAVIOR (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
