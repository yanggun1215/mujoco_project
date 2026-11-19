from pathlib import Path
import csv
import time
import mujoco
import mujoco.viewer
import numpy as np

TARGET_N = 10.0
RUN_SECONDS = 10.0
KP_FORCE = 0.6
KI_FORCE = 1.2
MAX_PRESS_N = 14.0
TORQUE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

scene_path = Path('models/franka_emika_panda/scene_torque_arm_only_box.xml')

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
mujoco.mj_forward(model, data)
assert model.nq == 7 and model.nv == 7 and model.nu == 7, '7 DoF / 7 torque motors expected'

site_id = model.site('press_tip').id
ft_site_id = model.site('ft_mount').id

sensor_face_id = model.geom('sensor_face').id
box_id = model.geom('press_box').id
box_top = float(model.geom_pos[box_id, 2] + model.geom_size[box_id, 2])

force_sensor = data.sensor('tip_force').data
torque_sensor = data.sensor('tip_torque').data
home_q = data.qpos.copy()
home_xyz = data.site_xpos[site_id].copy()

tool_mass = float(model.body_subtreemass[model.body('ft_tool').id])
tool_support_world = -tool_mass * model.opt.gravity

Jp = np.zeros((3, model.nv))
cf = np.zeros(6)
dt = model.opt.timestep
integral = 0.0
force_filtered = 0.0
rows = []


def contact_normal_force():
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if {int(c.geom1), int(c.geom2)} == {sensor_face_id, box_id}:
            mujoco.mj_contactForce(model, data, i, cf)
            total += max(0.0, float(cf[0]))  # compressive normal force [N]
    return total

print(f'Using fixed XML: {scene_path}')
print(f'Home tip z={home_xyz[2]:.3f} m | box top z={box_top:.3f} m | target={TARGET_N:.1f} N')
print('Force PI starts immediately; F/T feedback includes tool-weight compensation.')

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_print = 0.0
        while viewer.is_running() and data.time < RUN_SECONDS:
            t0 = time.perf_counter()
            q = data.qpos
            dq = data.qvel
            xyz = data.site_xpos[site_id]

            mujoco.mj_jacSite(model, data, Jp, None, site_id)
            vel = Jp @ dq
            R = data.site_xmat[ft_site_id].reshape(3, 3)
            sensor_world = R @ force_sensor
            contact_n = contact_normal_force()

            raw_n = max(0.0, float(tool_support_world[2] - sensor_world[2]))
            if contact_n < 0.03:
                raw_n = 0.0

            measured_n = raw_n

            error = TARGET_N - measured_n
            if contact_n >= 0.03:
                integral = float(np.clip(integral + error * dt, -4.0, 4.0))
            else:
                integral = 0.0
            down_cmd = float(np.clip(TARGET_N + KP_FORCE * error + KI_FORCE * integral,
                                     0.0, MAX_PRESS_N))

            fz = -down_cmd 

            force_cmd = np.zeros(3)
            force_cmd[:2] = np.clip(
                240.0 * (home_xyz[:2] - xyz[:2]) - 35.0 * vel[:2], -25.0, 25.0)
            force_cmd[2] = fz

            nullspace = np.eye(7) - Jp.T @ np.linalg.pinv(Jp.T, rcond=1e-4)
            tau_posture = nullspace @ (12.0 * (home_q - q) - 2.5 * dq)

            tau = data.qfrc_bias + Jp.T @ force_cmd + tau_posture
            data.ctrl[:] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

            rows.append([data.time, 'force_pi', TARGET_N, measured_n, contact_n, down_cmd,
                         *force_sensor, *sensor_world, *torque_sensor,
                         *xyz, *data.ctrl])
            if data.time >= next_print:
                print(f't={data.time:5.2f} s | '
                      f'F/T={measured_n:5.2f} N | contact={contact_n:5.2f} N | '
                      f'cmd={down_cmd:5.2f} N')
                next_print += 0.5

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
finally:
    output = Path('panda_box_pi_static_xml.csv')
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time_s', 'mode', 'target_N', 'FT_normal_N', 'contact_solver_normal_N',
            'command_down_N',
            'FT_Fx_local_N', 'FT_Fy_local_N', 'FT_Fz_local_N',
            'FT_Fx_world_N', 'FT_Fy_world_N', 'FT_Fz_world_N',
            'FT_Tx_local_Nm', 'FT_Ty_local_Nm', 'FT_Tz_local_Nm',
            'tip_x_m', 'tip_y_m', 'tip_z_m',
            *[f'tau_joint{i}_Nm' for i in range(1, 8)],
        ])
        writer.writerows(rows)

