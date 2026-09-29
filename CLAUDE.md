# BEHAVIOR Challenge 2026 — training & eval pipeline (team runbook)

Read this before doing any training, eval or task selection in this repo. It is the distilled
experience of the 2026-09 runs; the numbers below are measured, not guesses.

## What this is
- Policy: the 2025 winner's pi0.5-based model (Ilia Larchenko / Robot Learning Collective,
  https://github.com/IliaLarchenko/behavior-1k-solution, ~3B params, task-ID embeddings, no language),
  fine-tuned per task on the 2026 BEHAVIOR-1K demos. `behavior-1k-solution/` here is a snapshot of that
  repo with our 2026 changes (upstream commit `ca556f7`; see `patches/` for the exact diffs, including the
  `openpi` submodule and one BEHAVIOR-1K eval file).
- Training runs on a RunPod A100 80GB; eval runs locally on a 12 GB laptop GPU with OmniGibson.
- `pipeline/`: bootstrap (train), stream (pod -> local disk), eval and sweep scripts.
  `pipeline/pod/`: helpers that live in the pod's `/workspace` (`ckpt_sync.py`, ...).
- `data/goal_predicates_all_tasks.tsv` + `data/partial_credit_tasks.xlsx`: goal-predicate counts for all
  100 tasks (computed with the same BDDL grounding the official scorer uses).

## Scoring (from omnigibson/metrics/task_metric.py)
- Per instance: q = (goal predicates NEWLY satisfied) / (total ground goal predicates), max over grounding
  options; 1.0 on full success. Predicates already true at reset count in the denominator but never pay.
- Official task score = sum of q over the 20 public instances (301-320) / 20. Unrun instances count 0 —
  ALWAYS finish all 20. Overall = sum over 100 tasks / 100.

## Results so far (official q, 20/20 instances)
| Task | q | init | notes |
|---|---|---|---|
| 1 picking_up_trash | 0.5167 | winner checkpoint_2 | 5 solves; best task |
| 40 make_microwave_popcorn | 0.350 | winner checkpoint_4 | all-or-nothing in practice |
| 56 make_rose_centerpieces | 0.2625 | 50t generalist, joint 56/78/99 run | all partial credit |
| 89 installing_a_fax_machine | 0.250 | 50t generalist | best ckpt 8000, not 9999 |
| 10 set_up_a_coffee_station | 0.225 | winner checkpoint_1 | all partial |
| 42 chop_an_onion | 0.100 | winner checkpoint_1 | |
| 99 sorting_books_on_shelf | 0.018 | joint run | dedicated retrain being evaluated |
| 78, 46, 34, 63, 69 | 0 | | all-or-nothing or long horizon |

## How to pick a task (in this order)
1. **Check the team tracker first** (OneDrive sheet "Stanford BEHAVIOR Challenge: Task Tracker"). Never
   train a task someone else claimed. Never start training until the task is claimed.
2. **Partial credit**: >= 2 goal predicates that are false at reset and individually achievable
   (see `data/`). All-or-nothing tasks (0, 34, 35, 36, 37, 60, 63, 69, 78, 83, 93) scored 0 for us.
   Note: `not touching(x, floor)`-style goals ARE earnable when x starts on the floor.
3. **Short human demos**: everything we scored has demo length < ~6,500 steps (`length` in
   `BEHAVIOR-1K/datasets/2026-challenge-task-instances/metadata/task.jsonl`). 7.7k-step tasks scored ~0.
4. **Tasks 0-49: use the winner's per-task checkpoint** (ckpt1-4; map below) — our top two tasks did.
   The winner's own per-task 2025 scores (`BEHAVIOR-1K/docs/challenge_submissions/standard.{public,hidden}.
   Robot_Learning_Collective.*.json`) predict what is reachable (we kept 78% of it on task 1).
   Winner map: ckpt1 = 2,3,5,6,10,11,13,14,15,19,23,24,25,28,29,34,42,44,47,48; ckpt2 = 0,1,7,8,9,12,16,17,
   18,20,21,22,26,30,43,45; ckpt3 = 4,27,31,32,33,35,36,37,38,39,41,46,49; ckpt4 = 40.
   Tasks 50-99 start from the 50-task generalist (`IliaLarchenko/behavior_50t_checkpoint`).
5. Avoid slicing/cooking-heavy goals unless other goals pay independently (task 46 cook_bacon: winner 0.76,
   us 0).

## Training recipe (do not change without a reason)
- One task per run. 10k steps, **batch 16** (32 does not fit 80 GB), lr 5e-5, `--exp_name=ft1` ALWAYS
  (it is the last path component of the checkpoint dir; every script assumes it).
- Episodes: task N -> N*200 .. N*200+199; configs hold out the last 20.
- Specialist init (tasks 0-49): `num_tasks=NUM_TASKS_2025` (50) so the trained task embedding is kept.
  Generalist init (50-99): default 100 task slots.
- **Final checkpoint step = num_train_steps - 1 = 9999** (never 10000). Sweep steps 2000,4000,6000,8000,9999.
- Params-only checkpoints (`B1K_PARAMS_ONLY_CHECKPOINTS=1`): saving the 38 GB optimizer state OOM-killed the
  trainer. Consequence: no exact `--resume`; recovery = new run seeded from the newest params.
- ~9.2-9.5 GPU-hours per run (~3.2-3.4 s/step), ~$15-19.

## Pod workflow
1. Deploy a pod on the network volume (same datacenter). The Python venv lives on the container disk and
   is wiped on restart: `cp -r /workspace/behavior-1k-solution/.venv /root/venv_b1k` (needs /usr/bin/python3.11).
   A venv ON the network volume makes DataLoader workers take ~20 min to spawn.
2. Add a config in `behavior-1k-solution/src/b1k/training/config.py` (copy task 1 for specialist init,
   task 99/67 for generalist init), then copy a `pipeline/bootstrap_task<N>_training.sh` and run it with
   `POD=root@<ip> PORT=<port>` (direct TCP SSH, not the ssh.runpod.io proxy — rsync needs it).
   The bootstrap: rsync repo, download the task's dataset chunk (rgb only), prune non-rgb video keys, copy
   norm_stats + FAST tokenizer from the init checkpoint, one-batch sanity check (expect
   `tokenized_prompt [[N, s], ...]`), start `ckpt_sync.py` + training in tmux, start `stream_ckpts.sh`.
3. `pipeline/pod/ckpt_sync.py` copies sweep steps AND the newest step to the volume (crash insurance) and
   prunes the rest. `pipeline/stream_ckpts.sh` rsyncs sweep steps to the local disk, verifies byte counts,
   then deletes the pod copies.
4. Stop the pod when done. `runpodctl stop pod <id>` over SSH fails ("API key not found") unless you take
   the key from PID 1: `export RUNPOD_API_KEY=$(tr "\0" "\n" < /proc/1/environ | grep ^RUNPOD_API_KEY= | cut -d= -f2-)`.
   Verify by SSH becoming unreachable.

## Eval workflow (local)
- `pipeline/run_ft_eval.py --task <name> --task-id N --config <cfg> --ckpt-root <dir> --step <S> --run <tag>`
  runs a fresh policy server + simulator per instance and skips instances that already have a json, so
  reruns resume. Output: `~/eval_outputs/<task>_<tag>_step<S>/`. Use a distinct `--run` tag per training
  run so results of different runs never share a folder.
- Checkpoint choice: screen 5 instances per sweep checkpoint, then all 20 on the best (task 89: 8000 beat
  9999). Screening ranks, it does not measure.
- A rollout takes ~40-55 min at full GPU speed (time limit = 1.5 x demo length).
- Only use a streamed checkpoint after the streamer logged "step N: verified" (rsync writes straight into
  the final dir, so a manifest file alone is not proof).

## Pitfalls we paid for
- **12 GB laptop GPU is at its limit** (policy 6.8 GB + sim 4.1 GB). Any desktop app that touches the GPU
  (WeChat, Chrome, file manager) can OOM a rollout (`RESOURCE_EXHAUSTED` / `vkAllocateMemory`). Close
  them during evals. Retry loops must keep going until 20/20 real results.
- **Lid close suspends the laptop** unless a `systemd-inhibit --what=handle-lid-switch:sleep:idle` lock is
  held (see `pipeline/hold_awake.sh`; it must also cover the checkpoint streamer).
- **After suspend/resume the GPU can stay capped at 20 W / 180 MHz** (fan silent, sim ~9% CPU). Check
  `nvidia-smi --query-gpu=clocks.sm,power.draw,pstate --format=csv`; `sudo nvidia-smi -pl 60`, replug AC,
  or reboot between instances. Scores stay valid, only wall time suffers.
- **External NTFS drive unplugged uncleanly** -> mount refuses rw ("wrong fs type"): `sudo ntfsfix -d`.
  A vanished checkpoint dir makes every retry fail instantly — eval scripts must wait for the drive.
- `pkill -f <script name>` from an automated shell can kill that shell itself; kill by PID.
- The `!`-prefixed Claude Code prompt has no TTY: anything needing sudo must run in a real terminal.

## Secrets (never commit)
W&B key (`~/.netrc`, pod `/workspace/.netrc_wandb`), Hugging Face token, RunPod API key, SSH keys.
