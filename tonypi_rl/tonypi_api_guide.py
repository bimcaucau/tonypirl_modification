#!/usr/bin/env python3
"""
TonyPi MuJoCo Hardware & Software Access Guide.

This script demonstrates how to query, command, and inspect:
1. All 18 Joints (names, IDs, qpos indices, qvel indices, angles, limits)
2. All 18 Actuators (PD servo control data.ctrl, gains, transmission)
3. Sensors (IMU accelerometer, gyro, orientation quaternion, foot touches)
4. Bodies & Sites (pelvis position, velocity, rotation matrix, transformation)
5. Collision Contacts & Forces
"""

import os
import mujoco
import numpy as np

# Load scene model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scene_xml = os.path.join(base_dir, "tonypi_mujoco", "scene.xml")
model = mujoco.MjModel.from_xml_path(scene_xml)
data = mujoco.MjData(model)

# Forward kinematics to populate state quantities
mujoco.mj_forward(model, data)

print("=" * 80)
print("     TONYPI MUJOCO PYTHON ACCESS CHEAT SHEET")
print("=" * 80)

# =============================================================================
# 1. ACCESSING JOINTS
# =============================================================================
print("\n[1] ACCESSING JOINTS")
print("-" * 50)
print("Root freejoint (floating base) occupies:")
print("  - data.qpos[0:3] = root position [x, y, z] in world frame")
print("  - data.qpos[3:7] = root orientation quaternion [w, x, y, z]")
print("  - data.qvel[0:3] = root linear velocity [vx, vy, vz] in world frame")
print("  - data.qvel[3:6] = root angular velocity [wx, wy, wz] in world frame")
print("The 18 robot revolute joints occupy:")
print("  - data.qpos[7:25] (18 joint angles in radians)")
print("  - data.qvel[6:24] (18 joint angular velocities in rad/s)\n")

joint_names = [
    "neck_yaw", "head_pitch",
    "l_shoulder_pitch", "l_upperarm_roll", "l_elbow",
    "r_shoulder_pitch", "r_upperarm_roll", "r_elbow",
    "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
]

print(f"{'Idx':<4} {'Joint Name':<20} {'Joint ID':<10} {'qpos_adr':<10} {'qvel_adr':<10} {'Limits [rad]':<20}")
print("-" * 75)
for i, name in enumerate(joint_names):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_adr = model.jnt_qposadr[jnt_id]
    qvel_adr = model.jnt_dofadr[jnt_id]
    lim = model.jnt_range[jnt_id]
    print(f"{i:<4} {name:<20} {jnt_id:<10} {qpos_adr:<10} {qvel_adr:<10} [{lim[0]:+.3f}, {lim[1]:+.3f}]")

jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee")
knee_angle = data.qpos[model.jnt_qposadr[jnt_id]]
print(f"\n--> Example Reading: Left Knee Angle = {knee_angle:.4f} rad ({np.degrees(knee_angle):.1f} deg)")

# =============================================================================
# 2. ACCESSING ACTUATORS (COMMANDING SERVOS)
# =============================================================================
print("\n[2] ACCESSING ACTUATORS (COMMANDING SERVOS)")
print("-" * 50)
print("TonyPi uses 18 position actuators (<position joint=... kp=... kv=.../>).")
print("Actuator control array: data.ctrl of shape (18,)")
print("data.ctrl[i] sets target angle in RADIANS for actuator i.")
print("Torque is generated: tau = kp * (ctrl - qpos) - kv * qvel\n")

print(f"{'Act ID':<8} {'Actuator Name':<24} {'Target Joint':<20} {'Kp':<8} {'Kv':<8}")
print("-" * 70)
for i in range(model.nu):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trnid = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
    kp = model.actuator_gainprm[i, 0]
    kv = model.actuator_biasprm[i, 2] if model.actuator_biastype[i] != 0 else 0.0
    print(f"{i:<8} {act_name:<24} {jnt_name:<20} {kp:<8.1f} {abs(kv):<8.1f}")

print("\n--> How to send commands:")
print("    # 1. Set all 18 joints at once:")
print("    data.ctrl[:] = np.zeros(18)")
print("    # 2. Command single actuator by name:")
print("    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'l_knee_act')")
print("    data.ctrl[act_id] = 0.5  # 0.5 rad (~28.6 deg)")

# =============================================================================
# 3. ACCESSING SENSORS
# =============================================================================
print("\n[3] ACCESSING SENSORS")
print("-" * 50)
print(f"{'Sensor Name':<16} {'Type ID':<10} {'Dim':<6} {'Data Slice':<12} {'Current Values'}")
print("-" * 75)
for i in range(model.nsensor):
    sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    stype = model.sensor_type[i]
    dim = model.sensor_dim[i]
    adr = model.sensor_adr[i]
    val = data.sensordata[adr:adr+dim]
    val_str = np.array2string(val, precision=3, suppress_small=True)
    print(f"{sname:<16} {stype:<10} {dim:<6} [{adr}:{adr+dim}]      {val_str}")

print("\n--> Sensor Access Code Examples:")
print("    accel = data.sensor('imu_accel').data  # [ax, ay, az] in m/s^2")
print("    gyro  = data.sensor('imu_gyro').data   # [wx, wy, wz] in rad/s")
print("    quat  = data.sensor('imu_quat').data   # [w, x, y, z] orientation")
print("    l_touch = data.sensor('l_foot_touch').data[0] # contact force in N")

# =============================================================================
# 4. ACCESSING BODIES & SITES
# =============================================================================
print("\n[4] ACCESSING BODIES & SITES")
print("-" * 50)
pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
pelvis_pos = data.xpos[pelvis_id]
pelvis_rot = data.xmat[pelvis_id].reshape(3, 3)
print(f"Pelvis World Position: {pelvis_pos.round(4)}")
print(f"Pelvis Height (Z):     {pelvis_pos[2]:.4f} meters")
print(f"Pelvis Up-Vector:      {pelvis_rot[:, 2].round(4)}")
imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
print(f"IMU Site World Pos:    {data.site_xpos[imu_site_id].round(4)}")

# =============================================================================
# 5. ACCESSING CONTACTS
# =============================================================================
print("\n[5] ACCESSING CONTACTS")
print("-" * 50)
print(f"Active contact points currently: {data.ncon}")
for c_idx in range(min(data.ncon, 5)):
    c = data.contact[c_idx]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"  Contact {c_idx}: {g1} <-> {g2} at pos={c.pos.round(3)}")

print("\n" + "=" * 80)
print("     CHEAT SHEET EXECUTION COMPLETE")
print("=" * 80)
