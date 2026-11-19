from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

scene = Path('models/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
q0 = data.qpos[:7].copy()

points = np.array([[1.2, 0.1, 0.3, 0, 0, 0, 0],
                   [1, 0.5, 0.4, 0, 0, 0, 0],
                   [0.5, 0.7, 0.5, 0, 0, 0, 0]])
waypoints = np.vstack([q0, q0 + points])
segment_time = 3.0
lam, kd = 3.0, 20.0

inv_data = mujoco.MjData(model)

rows = []
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 10 : 
        t = float(data.time)
        k = min(int(t / segment_time), len(points) - 1)
        u = np.clip((t - k * segment_time) / segment_time, 0., 1.)
        a, b = waypoints[k], waypoints[k + 1]
        d = b - a
        q_des = a + (u*u - u*u*u)*d
        dq_des = (2*u - 3*u*u)*d / segment_time
        ddq_des = (2 - 6*u)*d / segment_time**2
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = q_des - q
        dq_ref = dq_des + lam*error
        ddq_ref = ddq_des + lam*(dq_des - dq)
        s = dq - dq_ref

        inv_data.qpos[:] = data.qpos
        inv_data.qvel[:] = data.qvel
        inv_data.qacc[:] = 0.0
        inv_data.time = data.time
        mujoco.mj_forward(model, inv_data)
        inv_data.qacc[:] = 0.0
        inv_data.qacc[:7] = ddq_ref
        mujoco.mj_inverse(model, inv_data)
        inverse_tau = inv_data.qfrc_inverse[:7].copy()
        tau = inverse_tau - kd*s
        data.ctrl[:7] = tau
        if step % 10 == 0:
            rows.append([t, k+1, *q_des, *q, *error, *tau])
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

header = ['time', 'segment'] + [f'{prefix}{i}' for prefix in ('target', 'q', 'error', 'tau') for i in range(1, 8)]

np.savetxt('pidc_data.csv', rows, delimiter=',', header=','.join(header), comments='')
r = np.asarray(rows)
plt.plot(r[:, 0], r[:, 2:5], '--', label='Target')
plt.plot(r[:, 0], r[:, 9:12], label='Actual')
plt.xlabel('Time (s)')
plt.ylabel('Joint (rad)')
plt.grid()
plt.legend()
plt.savefig('pidc_position.png')
plt.show()

