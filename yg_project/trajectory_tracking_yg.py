from pathlib import Path
import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

scene = Path('models/franka_emika_panda/scene_nohand.xml')
model = mujoco.MjModel.from_xml_path(str(scene))

data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

rows =[]
step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and data.time < 10:
        t = time.time() - start
        target = 0.3 * np.sin(t) 
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()

        data.ctrl[0] = target
        mujoco.mj_step(model, data)

        if step % 10 == 0:
            rows.append([t,target,q])

        viewer.sync()
        step +=1

    header = ['time'] + [f'{prefix}{i}' for prefix in ('target','q') for i in range(1,8)]
    r = np.asarray(rows)
    np.savetxt('trajectory_tracking.csv', r, delimiter=',', header=','.join(header), comments='')
    plt.plot(r[:,0],r[:,1], '--',label = 'Target')
    plt.plot(r[:,0],r[:,8],label = 'Actual')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Joint (rad)')
    plt.legend()
    plt.savefig('trajectory_tracking.png')
    # plt.show()



        
