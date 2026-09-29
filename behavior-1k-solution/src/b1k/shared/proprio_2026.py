"""R1Pro proprioception layouts for both BEHAVIOR Challenge harness generations.

* 2025 layout (256-dim): what ``omnigibson.learning.utils.eval_utils.PROPRIOCEPTION_INDICES`` described
  and what the 2025-challenge-demos LeRobot dataset stores in ``observation.state``. Some early
  2026 dataset exports (e.g. the local ``behavior-challenge-2026/datasets/hanging_pictures``) still use it.
* 2026 layout (61-dim): ``omnigibson/eval/utils/eval_utils.py`` in the 2026 BEHAVIOR-1K, i.e. what the
  2026 eval harness sends over the websocket AND what ``behavior-1k/2026-challenge-demos`` stores.

Both layouts use the same field names, so name-based indexing works for either; pick the layout by
the vector length with :func:`get_proprio_indices`. ``PROPRIOCEPTION_INDICES`` (the name the original
code imported from OmniGibson) is kept as an alias of the 2026 layout.

NOTE: in 2026 ``base_qvel`` is reported in the robot-local frame; in the 2025 data it was the raw
world-frame holonomic joint velocity. The base yaw is not exposed in the standard track, so this
cannot be converted at eval time.
"""
from collections import OrderedDict

import numpy as np

PROPRIOCEPTION_INDICES_2026 = {
    "R1Pro": OrderedDict(
        {
            "base_qvel": np.s_[0:3],
            "arm_left_qpos": np.s_[3:10],
            "arm_left_qvel": np.s_[10:17],
            "eef_left_pos": np.s_[17:20],
            "eef_left_quat": np.s_[20:24],
            "gripper_left_qpos": np.s_[24:26],
            "gripper_left_qvel": np.s_[26:28],
            "arm_right_qpos": np.s_[28:35],
            "arm_right_qvel": np.s_[35:42],
            "eef_right_pos": np.s_[42:45],
            "eef_right_quat": np.s_[45:49],
            "gripper_right_qpos": np.s_[49:51],
            "gripper_right_qvel": np.s_[51:53],
            "trunk_qpos": np.s_[53:57],
            "trunk_qvel": np.s_[57:61],
        }
    ),
}

PROPRIOCEPTION_INDICES_2025 = {
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[0:28],
            "joint_qpos_sin": np.s_[28:56],
            "joint_qpos_cos": np.s_[56:84],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],
            "robot_ori_cos": np.s_[143:146],
            "robot_ori_sin": np.s_[146:149],
            "robot_2d_ori": np.s_[149:150],
            "robot_2d_ori_cos": np.s_[150:151],
            "robot_2d_ori_sin": np.s_[151:152],
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "gripper_left_qpos": np.s_[193:195],
            "gripper_left_qvel": np.s_[195:197],
            "arm_right_qpos": np.s_[197:204],
            "arm_right_qpos_sin": np.s_[204:211],
            "arm_right_qpos_cos": np.s_[211:218],
            "arm_right_qvel": np.s_[218:225],
            "eef_right_pos": np.s_[225:228],
            "eef_right_quat": np.s_[228:232],
            "gripper_right_qpos": np.s_[232:234],
            "gripper_right_qvel": np.s_[234:236],
            "trunk_qpos": np.s_[236:240],
            "trunk_qvel": np.s_[240:244],
            "base_qpos": np.s_[244:247],
            "base_qpos_sin": np.s_[247:250],
            "base_qpos_cos": np.s_[250:253],
            "base_qvel": np.s_[253:256],
        }
    ),
}

PROPRIO_DIM_2025 = 256
PROPRIO_DIM_2026 = 61

# Alias kept for the original import sites (eval harness / 2026 data).
PROPRIOCEPTION_INDICES = PROPRIOCEPTION_INDICES_2026


def get_proprio_indices(proprio_dim: int, robot: str = "R1Pro") -> "OrderedDict[str, slice]":
    """Return the name->slice layout matching a proprio vector of the given length."""
    if proprio_dim == PROPRIO_DIM_2026:
        return PROPRIOCEPTION_INDICES_2026[robot]
    if proprio_dim == PROPRIO_DIM_2025:
        return PROPRIOCEPTION_INDICES_2025[robot]
    raise ValueError(
        f"Unrecognised R1Pro proprio vector length {proprio_dim}; expected {PROPRIO_DIM_2026} (2026 harness/"
        f"dataset) or {PROPRIO_DIM_2025} (2025 layout)."
    )
