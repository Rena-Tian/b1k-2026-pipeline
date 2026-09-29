"""Send synthetic 2026-harness observations to the winner policy server and check the reply."""
import sys, time, numpy as np
sys.path.insert(0, '/home/rena/behavior-1k-solution/src')
from b1k.shared.network_utils_2026 import WebsocketClientPolicy

c = WebsocketClientPolicy(host='127.0.0.1', port=8000)
rng = np.random.default_rng(0)
def obs():
    return {
        'robot_r1::proprio': rng.standard_normal(61).astype(np.float32),
        'robot_r1::robot_r1:zed_link:Camera:0::rgb': rng.integers(0, 255, (720, 720, 4), dtype=np.uint8),
        'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb': rng.integers(0, 255, (480, 480, 4), dtype=np.uint8),
        'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb': rng.integers(0, 255, (480, 480, 4), dtype=np.uint8),
        'robot_r1::cam_rel_poses': rng.standard_normal(21).astype(np.float32),
        'task_id': np.array([89], dtype=np.int64),
    }
c.reset()
t = time.time(); a = c.act(obs()); print('first act (incl. jit) %.1fs' % (time.time()-t), a.shape, a.dtype, a[:8].numpy().round(3))
for i in range(45):
    t = time.time(); a = c.act(obs()); dt = time.time()-t
    if dt > 0.5: print('step', i+1, 'act %.3fs' % dt, 'finite', bool(np.isfinite(a.numpy()).all()))
print('metadata', c.get_server_metadata())
print('SMOKETEST_OK')
