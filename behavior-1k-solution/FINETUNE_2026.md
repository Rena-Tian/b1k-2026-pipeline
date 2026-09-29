# Fine-tuning the 2025 winner's checkpoint on 2026-challenge tasks

Local port of https://github.com/IliaLarchenko/behavior-1k-solution to the 2026 BEHAVIOR Challenge
(2026 OmniGibson eval harness + `behavior-1k/2026-challenge-demos` LeRobot v3 data). See `git diff`
(and `git diff` inside `openpi/`) for every change; new modules are listed at the bottom.

## What changed vs the original repo

| Area | Change |
|---|---|
| Eval harness | `src/b1k/shared/network_utils_2026.py` (2026 websocket protocol), `proprio_2026.py` (61-dim 2026 / 256-dim 2025 proprio layouts, picked by vector length). `scripts/serve_b1k.py` and the wrappers import these instead of `omnigibson.learning.*`. |
| Model | `TASK_NUM_STAGES` extended to the 100 tasks of 2026 (`clamp(ceil(avg_demo_seconds*30/900), 5, 15)`, verified to reproduce the original 50 values). Model tables are sized from `config.num_tasks` via `stage_layout()`; a 50-task model is bit-identical to the original. |
| Weight loading | `PiBehaviorWeightLoader` restores through `openpi.models.model.restore_params` (multi-GPU checkpoints restore on any device set) and pads `task_embeddings` (50→100 rows) and `task_stage_embeddings` (596→1105 rows); new rows = mean of the trained rows + 10%·std noise. |
| Data | `dataset_format="lerobot_v3"` in `DataConfig`: `src/b1k/training/lerobot_v3.py` loads a LeRobot v3 root (official export or any per-task subset), maps the dataset's `task_index` to the 2026 task id from `meta/tasks` (names or instructions; `task_id_map` overrides), computes stage labels from per-episode lengths. Norm-stats / FAST-tokenizer scripts read v3 chunk files via `episode_files.py`. |
| Env | `lerobot` → wensi-ai fork `release/b1k` (v3 datasets), `transformers` 4.53 → 5.5.4 (dependency of that), FAST tokenizer loaded manually (`fast_tokenizer_loader.py`) because `AutoProcessor` breaks on transformers 5. Pinned openpi submodule: 2 small edits (pyproject pins, omnigibson import fallback). |
| Configs | `pi_behavior_b1k_2026_local34` (laptop smoke test), `pi_behavior_b1k_2026_task89` (pilot), `pi_behavior_b1k_2026_3more` (56/78/99) in `src/b1k/training/config.py`. |

Verified locally (laptop, 2026-09-20): data pipeline on the local hanging_pictures v3 subset → normalized
batch with `tokenized_prompt=[34, stage]`, FAST tokens, 224×224 images; winner checkpoint_1 loaded into
the 100-task model on CPU; one `compute_detailed_loss` pass on a real batch gives action_loss 0.08 /
FAST accuracy 0.82 (i.e. data alignment matches what the model was trained on). Full training does not
fit the 12 GB laptop GPU (train state alone is ~51 GB) — use the A100-80GB pod.

## Data caveats found

* `observation.state` may be **256-dim (2025 layout)** or **61-dim (2026 layout)** depending on the export;
  both handled by name-based indexing. `base_qvel` is world-frame in 2025 data and robot-frame in 2026
  data/eval; fine-tuning on 2026 data adapts the model to the 2026 convention.
* The local `~/behavior-challenge-2026/datasets/hanging_pictures` export had a **per-episode `index`
  column** (LeRobot v3 requires a global frame index) — the loader silently fetched wrong action
  windows. Repaired in place (`data_orig_bad_index/` keeps the original). Check any other subset with:
  `pd.read_parquet(f, columns=['index'])['index'].is_monotonic_increasing` across files.
* The official export uses task *names* in `meta/tasks` and sequential episode ids (200/task:
  task 89 = episodes 17800–17999).

## Pod recipe (pilot on task 89)

```bash
# 0. on the pod: clone this repo state (rsync ~/behavior-1k-solution minus .venv/outputs), then
cd behavior-1k-solution && GIT_LFS_SKIP_SMUDGE=1 uv sync

# 1. winner checkpoint (generalist 50-task init recommended for new tasks; ~12.6 GB)
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("IliaLarchenko/behavior_50t_checkpoint", local_dir="/workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint")
PY

# 2. assets: reuse the winner's normalisation stats + FAST tokenizer (keeps the scaling the weights expect)
A=outputs/assets/pi_behavior_b1k_2026_task89/b1k_2026_task89; mkdir -p $A
cp -r /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/

# 3. sanity: one batch through the pipeline (checks task_index->task id map, index column, shapes)
JAX_PLATFORMS=cpu uv run python -c "
import dataclasses; from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config('pi_behavior_b1k_2026_task89'), batch_size=2, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1))); print(obs.tokenized_prompt, act.shape)"

# 4. train (A100-80GB: start at batch 16; if OOM use --batch_size=8 and/or --ema_decay=None)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_behavior_b1k_2026_task89 --exp_name=ft1 --overwrite \
  --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --no-wandb_enabled

# 5. eval (2026 harness) with the fine-tuned checkpoint, e.g. on this laptop:
.venv/bin/python scripts/serve_b1k.py --port 8000 --task-id 89 policy:checkpoint \
  --policy.config pi_behavior_b1k_2026_task89 --policy.dir <checkpoint_dir>/<step>
# then the usual omnigibson.eval.eval client (see ~/winner-eval/run_task34_winner.py for the pattern)
```

Expectations: task 89 has a 2-predicate goal, so partial credit (q>0) is measurable; the LoRA pi05
baseline is flat 0.0. If q does not move after ~10k steps, stop before spending on 56/78/99.

## New modules
`src/b1k/shared/{network_utils_2026,proprio_2026,task_list_2026,fast_tokenizer_loader}.py`,
`src/b1k/training/{lerobot_v3,episode_files}.py`, `FINETUNE_2026.md` (this file).
