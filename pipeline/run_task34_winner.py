#!/usr/bin/env python3
"""Phase2-style eval (public_test instances 0-9 -> 301..310) of task 34 hanging_pictures
using the 2025 BEHAVIOR Challenge winner's checkpoint_1 (IliaLarchenko/behavior_submission,
covers 2025 task 34 == 2026 task 34). Mirrors ~/task-56-78-99-eval/run_phase2.py, but the
policy server is the winner's serve_b1k.py from ~/behavior-1k-solution (its own uv venv),
patched to speak the 2026 eval-harness protocol (see git diff there).
"""
import argparse
import json
import logging
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/rena')
SOLUTION = ROOT / 'behavior-1k-solution'
OG = ROOT / 'BEHAVIOR-1K/OmniGibson'
CHECKPOINT = Path('/media/rena/Elements/openpi_checkpoints/winner_larchenko/checkpoint_1')
PORT = 8000
TASK = 'hanging_pictures'
OUT_DIR = ROOT / 'eval_outputs' / f'{TASK}_winner_ckpt1'


def server_cmd():
    return [str(SOLUTION / '.venv/bin/python'), '-u', 'scripts/serve_b1k.py',
            '--port', str(PORT), '--task-id', '34',
            'policy:checkpoint', '--policy.config', 'pi_behavior_b1k_fast',
            '--policy.dir', str(CHECKPOINT)]


def eval_cmd(instances):
    return [str(ROOT / 'miniconda3/condabin/conda'), 'run', '--no-capture-output', '-n', 'behavior',
            'python', '-u', '-m', 'omnigibson.eval.eval', '--task-name', TASK,
            '--robot-config', 'omnigibson/eval/r1pro.yaml', '--mode', 'public_test',
            '--host', '127.0.0.1', '--port', str(PORT),
            '--instance-indices', *[str(i) for i in instances],
            '--num-rollouts', '1', '--env-wrapper', 'omnigibson.eval.wrappers.RGBDFullResWrapper',
            '--output-dir', str(OUT_DIR), '--write-video']


def status(name, **extra):
    value = {'status': name, 'updated': time.time(), **extra}
    tmp = OUT_DIR / 'status.tmp'
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(OUT_DIR / 'status.json')
    logging.info('%s %s', name, extra)


def summarize():
    rows = []
    for f in sorted((OUT_DIR / 'json').glob(f'{TASK}_*.json')) if (OUT_DIR / 'json').exists() else []:
        d = json.loads(f.read_text())
        rows.append({'instance': d['instance_id'], 'q': d['q_score']['final'],
                     'success': d['success'], 'steps': d['steps']})
    (OUT_DIR / 'summary.json').write_text(json.dumps(rows, indent=2))
    return rows


def run(instances, jax_cpu=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', PORT)) == 0:
            raise RuntimeError(f'Port {PORT} already occupied')
    env = os.environ.copy()
    env['DISPLAY'] = ':1'
    if jax_cpu:
        env['JAX_PLATFORMS'] = 'cpu'
    with (OUT_DIR / 'server.log').open('a') as server_log, (OUT_DIR / 'evaluation.log').open('a') as eval_log:
        server = subprocess.Popen(server_cmd(), cwd=SOLUTION, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            status('starting_policy_server', pid=server.pid)
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited; see {OUT_DIR}/server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/healthz', timeout=3) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy server did not become ready within 30 minutes')
                time.sleep(5)
            evaluator = subprocess.Popen(eval_cmd(instances), cwd=OG, env=env, stdout=eval_log, stderr=subprocess.STDOUT)
            status('evaluating', pid=evaluator.pid, instances=instances)
            while evaluator.poll() is None:
                summarize()
                time.sleep(30)
            rows = summarize()
            if evaluator.returncode != 0:
                raise RuntimeError(f'Evaluation incomplete (exit {evaluator.returncode}); see {OUT_DIR}/evaluation.log')
            status('completed', results=rows)
        except Exception as exc:
            status('failed', error=str(exc), results=summarize())
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--jax-cpu', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run(a.instances, jax_cpu=a.jax_cpu)
