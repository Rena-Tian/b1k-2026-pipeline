# Configs for tasks trained after 2026-10-04

The first push of this repo (`c68e527`, 2026-10-04) predates most of the runs that went into the submission. This
commit adds them:

- `behavior-1k-solution/src/b1k/training/config.py`: every PI_BEHAVIOR config used through the 10/16 deadline, plus the
  loader support they need (`lerobot_v3.py`, `data_loader.py`: skill-annotation stage labels via `stage_labels="skills"`,
  and skill-segment oversampling via `oversample_skills` / `oversample_factor`).
- `patches/behavior-1k-solution.vs-ca556f7.diff`: regenerated against upstream.
- `patches/openpi-pi05.wensi-ai.vs-0cc8e35.diff`: the plain openpi π0.5 configs (task 69), as a diff against
  `wensi-ai/openpi@0cc8e35` (a different openpi from the winner's submodule).

All PI_BEHAVIOR runs: batch 16, AdamW, params-only checkpoints, held-out demos = the last 20 of each task
(episodes `task*200 .. task*200+179`). "ckptN" = `IliaLarchenko/behavior_submission` `checkpoint_N`; those inits keep
the 50-task tables (`num_tasks=50`). Q = local self-eval on public instances 301-310, 1 rollout, sum / 10.

## Rena's submitted tasks

| Task | Submitted checkpoint (HF) | Config | Init | Steps (submitted step) | LR schedule | Stages / extras | Q |
|---|---|---|---|---|---|---|---|
| 4 can_meat | `Rena-Tian/b1k-2026-task4-ft1` | `pi_behavior_b1k_2026_task4_lowlr` | ckpt3 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills | 0.033 |
| 8 rearranging_kitchen_furniture + 9 putting_up_Christmas_decorations_inside | `Rena-Tian/b1k-2026-task8-9-ft1` (one model, both tasks) | `pi_behavior_b1k_2026_task8_9` | ckpt2 | 50,000 (49999, picked by screening 20k/30k/40k/49999 on 311-315) | cosine peak 5e-5 (default warmup/decay) | skills; joint data from both tasks | 0.125 / 0.467 |
| 14 carrying_in_groceries | `Rena-Tian/b1k-2026-task14-ft1` | `pi_behavior_b1k_2026_task14_lowlr` | ckpt1 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills | 0.250 |
| 16 moving_boxes_to_storage | `Rena-Tian/b1k-2026-task16-ft1` | `pi_behavior_b1k_2026_task16_lowlr` | ckpt2 | 3,000 (2999) | cosine 5e-6 → 1e-6, warmup 100 | skills | 0.500 |
| 21 collecting_childrens_toys | `Rena-Tian/b1k-2026-task21-ft1` | `pi_behavior_b1k_2026_task21_lowlr` | ckpt2 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills | 0.286 |
| 26 assembling_gift_baskets | `Rena-Tian/b1k-2026-task26-ft1` | `pi_behavior_b1k_2026_task26` | ckpt2 | 10,000 (9999, screened vs 8000) | cosine peak 5e-5 | time stages (pre-10/11 run) | 0.269 |
| 43 slicing_vegetables | `Rena-Tian/b1k-2026-task43-ft1` | `pi_behavior_b1k_2026_task43_skills` | ckpt2 | 20,000 (19999, screened) | cosine peak 5e-5 | skills | 0.122 |
| 45 cook_hot_dogs | `Rena-Tian/b1k-2026-task45-ft1` | `pi_behavior_b1k_2026_task45` → `..._task45_r13k` → `..._task45_r13k5` | ckpt2 | 20,000 total (local 6499 of r13k5 = global 19999) | cosine peak 5e-5 → 5e-6 over 20k, run as warm restarts (see below) | skills | 0.600 |
| 69 vacuuming_floors | `Rena-Tian/b1k-2026-task69-ft1` | openpi `pi05_b1k_3task_calib`, run `run1` | `pi05_base` (LoRA) | 50,000 (49999) — config says 30k, see below | cosine 2.5e-5 → 2.5e-6 over 30k, then flat 2.5e-6 | π0.5, prompt = task slug | 0.000 |

Notes:

- **4, 14, 21** come from one list comprehension (`for _t in (15, 25, 29, 23, 48, 4, 36, 37, 21, 24, 14, 46)`); the
  init is chosen per task (`checkpoint_3` for 4/36/37/46, `checkpoint_2` for 21, `checkpoint_1` otherwise). Same recipe
  as `task16_lowlr` but 2,000 steps instead of 3,000.
- **16**: the low-LR run replaced the 5-task 10k run (`pi_behavior_b1k_2026_task7_16_18_30_43`, peak 5e-5), which dropped
  task 16 from 0.60 (winner ckpt2 unchanged) to 0.25. `Rena-Tian/b1k-2026-task16-winner-checkpoint2` is only a mirror of
  the unchanged winner weights, kept for reference; the submitted model is `task16-ft1`.
- **26** is part of the 10/08 batch comprehension (`(26, "checkpoint_2", 10_000, 16)`) and is the one submitted run that
  still used time-sliced stages. The factory default changed to `stage_labels="skills"` on 10/11, so to reproduce
  this run exactly, pass `stage_labels="time"`.
- **43**: the earlier `pi_behavior_b1k_2026_task43` (10k, time stages) scored 0 on screening; the submitted one is
  `task43_skills` (20k, skill stages, same init).
- **45**: the 20k run (`pi_behavior_b1k_2026_task45`) deadlocked in an orbax save at step 13500. Checkpoints are
  params-only, so it was continued as new runs from saved params: `task45_r13k` from global 13000 (7,000 steps,
  lr 1.784e-5 → 5e-6), which deadlocked again, then `task45_r13k5` from global 13500 (6,500 steps, warmup 50,
  lr 1.773e-5 → 5e-6, `--save_interval=2500`). Both continue the original cosine curve, so the whole thing
  approximates one 20k run. To reproduce, set `B1K_INIT_PARAMS` to the previous stage's params.
- **69**: `pi05_b1k_3task_calib` trains one LoRA π0.5 (`gemma_2b_lora` + `gemma_300m_lora`, action horizon 32,
  batch 96) on tasks 34/63/69 jointly, from `pi05_base`. The config sets `num_train_steps=30_000`; the run reached
  29999 and was then resumed with
  `train_b1k.py pi05_b1k_3task_calib --exp_name=run1 --resume --num_train_steps=50000`. The LR schedule was not
  changed, and openpi's default `CosineDecaySchedule` decays over 30k steps, so steps 30k-50k ran flat at 2.5e-6.
  The diff also renames the R1Pro robot to `robot_r1` / switches to the 2026 dataset camera keys, and registers
  the task slugs as prompts. Both are needed to train and serve it.

## Richard's submitted tasks

These were trained from configs in this same `config.py`. Config names come from the server logs of each task's
official 301-310 eval; steps come from the submitted checkpoint. All HF repos are under `RichardCTian/`.

| Task | Submitted checkpoint (HF) | Config | Init | Steps (submitted step) | LR schedule | Stages / extras |
|---|---|---|---|---|---|---|
| 7 | `RichardCTian/b1k-2026-task7-ft1` | `pi_behavior_b1k_2026_task7_lowlr` | ckpt2 | 3,000 (2999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 11 | `RichardCTian/b1k-2026-task11-ft1` | `pi_behavior_b1k_2026_task11_lowlr` | ckpt1 | 3,000 (2999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 15 | `RichardCTian/b1k-2026-task15-ft1` | `pi_behavior_b1k_2026_task15_lowlr` | ckpt1 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 17 | `RichardCTian/b1k-2026-task17-ft1` | `pi_behavior_b1k_2026_task17_lowlr` | ckpt2 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 18 | `RichardCTian/b1k-2026-task18-ft1` | `pi_behavior_b1k_2026_task18_lowlr` | ckpt2 | 3,000 (2999) | cosine 5e-6 → 1e-6, warmup 100 | skills; hard skills oversampled x3 |
| 24 | `RichardCTian/b1k-2026-task24-ft1` | `pi_behavior_b1k_2026_task24_lowlr` | ckpt1 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 25 | `RichardCTian/b1k-2026-task25-ft1` | `pi_behavior_b1k_2026_task25_lowlr` | ckpt1 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 29 | `RichardCTian/b1k-2026-task29-ft1` | `pi_behavior_b1k_2026_task29_lowlr` | ckpt1 | 2,000 (1999) | cosine 5e-6 → 1e-6, warmup 100 | skills |
| 30 | `RichardCTian/b1k-2026-task30-ft1` | `pi_behavior_b1k_2026_task30_lowlr` | ckpt2 | 3,000 (2999) | cosine 5e-6 → 1e-6, warmup 100 | skills; hard skills oversampled x3 |
| 32 | `RichardCTian/b1k-2026-task32-ft1` | `pi_behavior_b1k_2026_task32` | ckpt3 | 50,000 (40000, picked by screening 20k/30k/40k on 311-315) | see config | see config |
| 64 | `RichardCTian/b1k-2026-task64-ft1` | `pi_behavior_b1k_2026_task64` → `pi_behavior_b1k_2026_task64_cont` | 50t generalist | 20,000 + 30,000 (local 29999 of `task64_cont`) | see configs | 100-task model |

Notes:

- **15, 24, 25, 29** come from the same list comprehension as Rena's 4/14/21 (2,000 steps; init per task).
- **18, 30** come from the oversampling comprehension: skill segments `pick up from` / `place on next to` / `place on`
  (18) and `pick up from` / `turn on switch` / `ignite` / `turn off switch` (30) are repeated 3x.
- **Not submitted**: the group runs `task7_16_18_30_43`, `task7_18_30` and `task11_13_15_17_25` and the `_s500` runs
  all scored 0 on their screens.
- **32**: step 40000 had the best screen on 311-315 (mean 0.40, vs 0.10 at both 20000 and 30000), so the
  submitted checkpoint is not the run's final step.
- **64**: `task64` (20k from `IliaLarchenko/behavior_50t_checkpoint`) was continued for 30k steps by `task64_cont`
  (`B1K_INIT_PARAMS` = `ft_task64/19999` params; norm stats and tokenizer from the 50t checkpoint).
- **54**: `pi_behavior_b1k_2026_task54` (50t generalist init, 10,000 steps, batch 16, peak lr 5e-5, episodes
  10800-10979), trained on Richard's pod 2026-10-09 with `pipeline/bootstrap_task54_training.sh` (exp `ft1`).
