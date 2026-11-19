import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

scene = Path('models/franka_emika_panda/scene_torque_nohand.xml')
model = mujoco.MjModel.from_xml_path(str(scene))

data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

points = np.array([[0.50, 0, 0, 0, 0, 0, 0],[0.50, 0.25, 0, 0, 0, 0, 0],[0.20, -0.25, 0.45, 0, 0, 0, 0]])

segment_time = 3
kp = 100
kd = 50

q0 = data.qpos[:7].copy()
waypoints = q0 + points
rows = []
step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 10:
        k = min(int(data.time / segment_time), len(points) - 1)
        target = waypoints[k]
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = target - q
        tau = kp * error - kd * dq + data.qfrc_bias[:7]
        data.ctrl[:7] = tau

        if step % 10 == 0:
            rows.append([data.time, k + 1, *target, *q, *error, *tau])
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

header = ['time', 'point'] +[f'{prefix}{i}' for prefix in ('target', 'q', 'error', 'tau') for i in range(1, 8)]
np.savetxt('panda_setpoint_data.csv', rows, delimiter=',', header=','.join(header), comments='')

r = np.array(rows)
plt.plot(r[:, 0], r[:, 2:5], '--', label='Target')
plt.plot(r[:, 0], r[:, 9:12], label='Actual')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Joint (rad)')
plt.legend()
plt.savefig('setpoint_control.png')
plt.show()
