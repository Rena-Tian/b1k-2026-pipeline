#!/usr/bin/env python3
"""Evaluate a fine-tuned (winner-recipe) checkpoint on one 2026 task with the 2026 harness on this laptop.
Generalised from run_task89_ft_eval.py. Usage:
    run_ft_eval.py --task sorting_books_on_shelf --task-id 99 --config pi_behavior_b1k_2026_3more \
        --ckpt-root /media/rena/Elements/openpi_checkpoints/ft_3more --step 4000 [--instances 0 1 2 3 4]
Output: ~/eval_outputs/<task>_<run>_step<step>/ (status.json, summary.json, json/, videos/)
"""
import argparse, json, logging, os, socket, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path('/home/rena')
SOLUTION = ROOT / 'behavior-1k-solution'
OG = ROOT / 'BEHAVIOR-1K/OmniGibson'
PORT = 8000
TASK = None; TASK_ID = None; CONFIG = None; RUN = 'ft1'  # set from CLI in main


def server_cmd(ckpt_dir):
    return [str(SOLUTION / '.venv/bin/python'), '-u', 'scripts/serve_b1k.py', '--port', str(PORT),
            '--task-id', str(TASK_ID), 'policy:checkpoint', '--policy.config', CONFIG, '--policy.dir', str(ckpt_dir)]


def eval_cmd(instances, out_dir):
    return [str(ROOT / 'miniconda3/condabin/conda'), 'run', '--no-capture-output', '-n', 'behavior',
            'python', '-u', '-m', 'omnigibson.eval.eval', '--task-name', TASK,
            '--robot-config', 'omnigibson/eval/r1pro.yaml', '--mode', 'public_test',
            '--host', '127.0.0.1', '--port', str(PORT), '--instance-indices', *[str(i) for i in instances],
            '--num-rollouts', '1', '--env-wrapper', 'omnigibson.eval.wrappers.RGBDFullResWrapper',
            '--output-dir', str(out_dir), '--write-video']


def summarize(out_dir):
    rows = []
    jd = out_dir / 'json'
    if jd.exists():
        for f in sorted(jd.glob(f'{TASK}_*.json')):
            d = json.loads(f.read_text())
            rows.append({'instance': d['instance_id'], 'q': d['q_score']['final'], 'success': d['success'], 'steps': d['steps']})
    (out_dir / 'summary.json').write_text(json.dumps(rows, indent=2))
    return rows


def status(out_dir, name, **extra):
    v = {'status': name, 'updated': time.time(), **extra}
    (out_dir / 'status.tmp').write_text(json.dumps(v, indent=2)); (out_dir / 'status.tmp').replace(out_dir / 'status.json')
    logging.info('%s %s', name, extra)


def _serve_and_eval(ckpt_dir, instances, out_dir, step):
    """One policy-server + one evaluator process for the given instances."""
    env = os.environ.copy(); env['DISPLAY'] = ':1'
    # Policy on CPU unless POLICY_DEVICE=gpu: the 12 GB laptop GPU cannot hold sim + policy once the
    # desktop (Firefox/Thunderbird/Xorg, ~1 GB) is busy; CPU inference costs ~7.5 s per prediction.
    if os.environ.get('POLICY_DEVICE', 'cpu') == 'cpu':
        env['JAX_PLATFORMS'] = 'cpu'
    else:
        # Leave JAX's allocator at its defaults. Both tuning attempts made things worse on this
        # 12.2 GB card: ALLOCATOR=platform fragmented (task 40 step 4000 died 5/5 on a 210 MiB
        # request) and capping MEM_FRACTION=0.58 left too small a pool for the first jitted
        # inference call (task 46 died the same way). The defaults are what completed tasks 89 and
        # 56 at 20/20, so the retry passes in the sweep scripts are the right safety net instead.
        for k in ('XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR'):
            env.pop(k, None)
    with (out_dir / 'server.log').open('a') as slog, (out_dir / 'evaluation.log').open('a') as elog:
        server = subprocess.Popen(server_cmd(ckpt_dir), cwd=SOLUTION, env=env, stdout=slog, stderr=subprocess.STDOUT)
        try:
            status(out_dir, 'starting_policy_server', pid=server.pid, step=step, instances=instances)
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited; see {out_dir}/server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/healthz', timeout=3) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy server not ready within 30 minutes')
                time.sleep(5)
            evaluator = subprocess.Popen(eval_cmd(instances, out_dir), cwd=OG, env=env, stdout=elog, stderr=subprocess.STDOUT)
            status(out_dir, 'evaluating', pid=evaluator.pid, step=step, instances=instances)
            while evaluator.poll() is None:
                summarize(out_dir); time.sleep(30)
            return evaluator.returncode
        finally:
            if server.poll() is None:
                server.terminate()
                try: server.wait(timeout=20)
                except subprocess.TimeoutExpired: server.kill()
            time.sleep(10)  # let the GPU memory drain before the next pair


def run(step, instances, ckpt_root):
    """Evaluate one instance per (server, evaluator) pair. The 12 GB laptop GPU is too tight to keep
    the sim + policy alive across scene reloads (instance 2 OOMed at step 500), so each instance gets
    fresh processes; already-finished instances (json present) are skipped, so reruns resume."""
    ckpt_dir = Path(ckpt_root) / str(step)
    assert (ckpt_dir / 'params').exists(), f'no params in {ckpt_dir}'
    out_dir = ROOT / 'eval_outputs' / f'{TASK}_{RUN}_step{step}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', PORT)) == 0:
            raise RuntimeError(f'Port {PORT} already occupied')
    failures = []
    for inst in instances:
        done = {json.loads(f.read_text())['instance_id'] for f in (out_dir / 'json').glob(f'{TASK}_*.json')} if (out_dir / 'json').exists() else set()
        # public_test index i -> instance id 301+i (holds for every task evaluated so far: ids 301-310)
        if 301 + inst in done:
            logging.info('instance index %d already done, skipping', inst); continue
        try:
            rc = _serve_and_eval(ckpt_dir, [inst], out_dir, step)
            if rc != 0:
                failures.append(inst)
        except Exception as exc:
            logging.exception('instance %d failed: %s', inst, exc); failures.append(inst)
    rows = summarize(out_dir)
    status(out_dir, 'completed' if not failures else 'completed_with_failures', step=step, results=rows, failed_instances=failures)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True)  # int step, or a named checkpoint dir (e.g. checkpoint_4)
    ap.add_argument('--instances', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='ft1')
    a = ap.parse_args()
    TASK, TASK_ID, CONFIG, RUN = a.task, a.task_id, a.config, a.run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    print(run(a.step, a.instances, a.ckpt_root))
