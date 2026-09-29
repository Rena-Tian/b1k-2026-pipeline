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
- **Leaderboard task score = sum of q over public instances 0-9 (instance ids 301-310), ONE rollout each,
  default 1.5x timeouts, / 10.** Unrun instances count 0 — always finish all 10. Overall = sum over 100 tasks
  / 100 (a full submission is 100 tasks x 10 instances). Source: `BEHAVIOR-1K/docs/challenge/evaluation.md`
  and `submission.md` (2026 docs).
- Public instances 10-19 (311-320) do NOT count; they are the held-out test set. The organizers evaluate the
  top 5 again on 10 hidden instances.
- No cherry-picking: report the prescribed single rollout per instance. Re-running an instance that crashed
  without producing a result (e.g. GPU OOM) is fine; re-running a finished one to get a better score is not.
- History: until 2026-09-29 we averaged all 20 public instances (/20); that under-reported us (0.0172 vs 0.0199).

## Results so far (leaderboard method: instances 301-310, / 10; as of 2026-09-29)
| Task | leaderboard q | all-20 avg (old method) | checkpoint / init | notes |
|---|---|---|---|---|
| 1 picking_up_trash | **0.5667** | 0.5167 | ft_task1/9999, winner checkpoint_2 | 2 of 10 solved; best task |
| 40 make_microwave_popcorn | 0.400 | 0.350 | ft_task40/9999, winner checkpoint_4 | all-or-nothing in practice |
| 56 make_rose_centerpieces | 0.350 | 0.2625 | ft_3more/19999, 50t generalist, joint 56/78/99 | all partial credit |
| 89 installing_a_fax_machine | 0.250 | 0.250 | ft_task89/8000, 50t generalist | best ckpt 8000, not 9999 |
| 10 set_up_a_coffee_station | 0.2333 | 0.225 | ft_task10/9999, winner checkpoint_1 | all partial |
| 42 chop_an_onion | 0.150 | 0.100 | ft_task42/9999, winner checkpoint_1 | |
| 99 sorting_books_on_shelf | 0.0364 | — | ft_task99/9999 dedicated retrain, 50t generalist | old joint ckpt 0.018; public on HF `Rena-Tian/b1k-2026-task99-ft1` |
| 78, 46, 34, 63, 69 | 0 | 0 | | all-or-nothing or long horizon |
| 67 boxing_food_after_dinner | — | — | ft_task67 (trained 2026-09-29) | not yet evaluated |

Sum 1.9864 -> overall **0.0199**.

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
   Robot_Learning_Collective.*.json`) predict what is reachable (we kept 85% of it on task 1: 0.567 vs 0.667, both on instances 0-9).
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
- **Checkpoint choice on the TEST instances, score on the LEADERBOARD instances**: screen each sweep
  checkpoint (e.g. 6000, 8000, 9999) on a few of instances 10-19 (`--instances 10 11 12 13 14`, ids 311-315),
  pick the best (ties -> later step), then run instances 0-9 (301-310) once on that checkpoint and report
  sum/10. Selecting on 301-310 itself inflates the reported number. The last checkpoint is not always best
  (task 89: 8000 beat 9999); task 99's retrain improved at every checkpoint. Screening ranks, it does not measure.
- A full task eval is therefore 10 rollouts (+ screens), not 20.
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
