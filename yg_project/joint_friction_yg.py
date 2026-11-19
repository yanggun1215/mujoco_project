import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

scene = Path('models/franka_emika_panda/scene_torque.xml')
model = mujoco.MjModel.from_xml_path(str(scene))

model.dof_frictionloss[:7] = 0
model.dof_damping[:7] = 0

fc_n = np.array([-15.876, -8.3559, -3.0788, -2.6335, -2.2146, -1.8977, -0.1])
fc_p = np.array([15.693, 10.643, 2.6631, 2.4529, 2.1265, 1.8386, 0.1])
fv1_n = np.array([-103.03, -71.122, -4.943, -4.1702, -2.0895, -5.9460, -3])
fv1_p = np.array([106.10, 86.132, 10.142, 4.1354, 2.2052, 5.4817, 3])
fv2_n = np.array([-0.6523, 0.5446, -0.7078, -0.9260, 0.5379, -2.5524, -1])
fv2_p = np.array([0.6846, 0.6741, -0.6764, -0.7183, 0.6657, -2.0797, 1])
weight = np.array([1, 1, 1, 0.5, 0.2, 0.01, 0.01])

def friction_torque(dq):
    positive = dq > 0
    fc = np.where(positive, fc_p, fc_n)
    fv1 = np.where(positive, fv1_p, fv1_n)
    fv2 = np.where(positive, fv2_p, fv2_n)
    return weight * (fc * np.tanh(100 * np.abs(dq)) + fv1 * (1 - np.exp(-np.abs(dq / fv2))))

rows = []
results = []
for on in [0, 1]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    target = data.qpos[:7].copy()
    target[0] += 1.0
    target[0] = min(target[0], model.jnt_range[0, 1] - 0.05) if model.jnt_limited[0] else target[0]
    print("Friction", "ON" if on else "OFF")

    kp = 100
    kd = 20

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 4:
            q = data.qpos[:7].copy()
            dq = data.qvel[:7].copy()
            error = target - q
            tau = kp * error - kd * dq + data.qfrc_bias[:7]
            f = friction_torque(dq) if on else np.zeros(7)

            data.ctrl[:7] = tau
            data.qfrc_applied[:7] = -f
            rows.append([on, data.time, q[0], target[0], error[0], dq[0], tau[0], f[0], np.dot(-f, dq)])
            mujoco.mj_step(model, data)
            viewer.sync()

    r = np.array([x for x in rows if x[0] == on])
    results.append(r)
    print(('ON ' if on else 'OFF'), 'RMSE:', round(np.sqrt(np.mean(r[:, 4] ** 2)), 4),'Final error:', round(abs(r[-1, 4]), 4))

header = 'friction_on,time,q1,target_q1,error_q1,dq1,command_tau1,friction_tau1,friction_power'
np.savetxt('joint_friction_yg.csv', rows, delimiter=',', header=header, comments='')



