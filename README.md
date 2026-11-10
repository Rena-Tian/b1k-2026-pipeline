# b1k-2026-pipeline

Fine-tuning and evaluation pipeline for the BEHAVIOR Challenge 2026, built on the 2025 winning solution
([IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution), Apache-2.0).

- `CLAUDE.md` — the runbook: task selection rules, training recipe, pod + eval workflow, pitfalls, results.
- `behavior-1k-solution/` — snapshot of the winner's code with our 2026 changes (LeRobot v3 loader,
  2026 task configs, params-only checkpoints, ...). Upstream `ca556f7`; exact diffs in `patches/`.
- `pipeline/` — train bootstrap, checkpoint streaming, eval runner and sweep scripts; `pipeline/pod/` runs
  inside the pod.
- `CONFIGS_AFTER_2026-10-04.md` — which config, init, steps and LR each post-10/04 submitted checkpoint used.
- `data/` — goal-predicate / partial-credit table for all 100 tasks, and the team scoreboard page source.

Checkpoints and datasets are not in git (12.6 GB per checkpoint). Secrets are never committed.
