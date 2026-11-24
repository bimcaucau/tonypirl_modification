# Hiwonder TonyPi 18-DOF Reinforcement Learning (RL) Guide

This package provides a complete Gymnasium bipedal locomotion environment and training pipeline for the **18-DOF Hiwonder TonyPi humanoid robot** in **MuJoCo 3.x**.

---

## 1. Quick Start

Ensure you use your Python environment containing `gymnasium`, `stable_baselines3`, `torch`, and `mujoco`:

```bash
# Set Python path to project root
export PYTHONPATH="/mnt/d/robotics_cccc/tonypi:$PYTHONPATH"
PYTHON="/home/bimcaucau/miniconda3/bin/python"

# 1. Run the Interactive Hardware/Software Access Guide
$PYTHON tonypi_rl/tonypi_api_guide.py

# 2. Train with PPO (Parallelized across 4 environments)
$PYTHON tonypi_rl/train.py --timesteps 500000 --n_envs 4 --action_scale 0.25

# 3. Evaluate and Visualize Trained Policy in MuJoCo 3D Viewer
$PYTHON tonypi_rl/eval.py --model tonypi_rl/models/<RUN_NAME>/tonypi_final_model.zip --gui
```

---

## 2. Action Normalization & Action Scaling Design

### User Requirement:
> *"actions should be normalized between [-1, 1], and control all 18 joints of the robot, but the output action should be multiplied by pi or somewhat like that, not sending signal from [-1, 1] to the joints, I will re-design it later"*

### Implementation Location:
All action mapping logic is isolated inside [`tonypi_rl/tonypi_env.py`](file:///mnt/d/robotics_cccc/tonypi/tonypi_rl/tonypi_env.py) in the method `action_to_joint_targets(self, action: np.ndarray)`:

```python
def action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    
    if self.action_mode == "pi_scaled_offset":
        # Multiplied by pi with a scale factor around default standing pose:
        scale = self.action_scale * np.pi
        targets = self.default_standing_qpos + action * scale
        
    elif self.action_mode == "pi_direct":
        # Direct angle proportional to pi:
        scale = self.action_scale * np.pi
        targets = action * scale
        
    elif self.action_mode == "joint_limits":
        # Full joint limit mapping:
        targets = self.joint_limits_lower + 0.5 * (action + 1.0) * (
            self.joint_limits_upper - self.joint_limits_lower
        )
    
    # Clipped to physical joint limits
    return np.clip(targets, self.joint_limits_lower, self.joint_limits_upper)
```

### How to Re-Design It:
1. Open [`tonypi_rl/tonypi_env.py`](file:///mnt/d/robotics_cccc/tonypi/tonypi_rl/tonypi_env.py#L143-L178).
2. Modify `action_to_joint_targets()` to implement your custom kinematics, joint-specific multipliers (e.g. smaller scale for roll vs pitch), or trajectory generator.
3. You can also pass custom `--action_mode` and `--action_scale` directly from the command line in `train.py`.

---

## 3. How to Access Robot Hardware & Sensors in Python

### A. Accessing Joints (Angles & Speeds)

MuJoCo stores positions in `data.qpos` and velocities in `data.qvel`.
Because TonyPi has a floating base (`<freejoint name="root"/>`), the indices are structured as follows:

| Index Slice | Name | Description |
| :--- | :--- | :--- |
| `data.qpos[0:3]` | Root Position | Floating base XYZ in world frame ($m$) |
| `data.qpos[3:7]` | Root Quaternion | Base orientation $[w, x, y, z]$ |
| `data.qvel[0:3]` | Root Lin. Vel. | Linear velocity $[v_x, v_y, v_z]$ ($m/s$) |
| `data.qvel[3:6]` | Root Ang. Vel. | Angular velocity $[\omega_x, \omega_y, \omega_z]$ ($rad/s$) |
| `data.qpos[7:25]`| 18 Joint Positions | Joint angles in radians |
| `data.qvel[6:24]`| 18 Joint Velocities| Joint angular velocities in rad/s |

#### Reading joint angles in Python:
```python
# Read all 18 joints as a dictionary:
joint_positions = env.get_joint_positions()
print(joint_positions["l_knee"])

# Or directly from MuJoCo data by name:
jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee")
qpos_idx = model.jnt_qposadr[jnt_id]
knee_angle_rad = data.qpos[qpos_idx]
```

### B. Accessing Actuators (Commanding Servos)

TonyPi has 18 PD position actuators in `scene.xml` (`model.nu = 18`):
```python
# Actuator array has shape (18,) matching the 18 joints:
# Send target angles in RADIANS:
data.ctrl[:] = np.zeros(18)  # Standing zero pose

# Command a specific actuator by name:
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "l_knee_act")
data.ctrl[act_id] = 0.523  # 30 degrees in radians
```

### C. Accessing Sensors

Sensors defined in `tonypi.xml` are accessible through `data.sensor(name).data`:

```python
# 1. Torso IMU Accelerometer (m/s^2)
imu_accel = data.sensor("imu_accel").data  # np.ndarray shape (3,) -> [ax, ay, az]

# 2. Torso IMU Gyroscope (rad/s)
imu_gyro = data.sensor("imu_gyro").data    # np.ndarray shape (3,) -> [wx, wy, wz]

# 3. Torso Orientation Quaternion [w, x, y, z]
imu_quat = data.sensor("imu_quat").data    # np.ndarray shape (4,)

# 4. Foot Touch Sensors (Virtual contact normal force in Newtons)
l_touch = data.sensor("l_foot_touch").data[0]
r_touch = data.sensor("r_foot_touch").data[0]
```

### D. Accessing Bodies & Sites (Pose & Rotation)

```python
pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

# World Cartesian Position [x, y, z]:
pos = data.xpos[pelvis_id]

# 3x3 World Rotation Matrix:
rot_mat = data.xmat[pelvis_id].reshape(3, 3)

# Upright check (dot product of torso up-vector with world Z [0, 0, 1]):
uprightness = rot_mat[2, 2]  # 1.0 when perfectly upright, 0.0 when on side, <0 when upside-down

# Site World Position:
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
site_pos = data.site_xpos[site_id]
```

---

## 4. Environment Observation Space (Default: 63 Dimensions)

By default, `include_lin_vel=False` and `include_foot_contacts=False` to strictly match the **real physical TonyPi hardware**:

| Slices | Dimension | Feature | Physical Source on Robot |
| :--- | :--- | :--- | :--- |
| `obs[0:3]` | 3 | Projected Gravity | Computed from IMU accelerometer ($R_{\text{body}}^T \cdot [0, 0, -1]^T$) |
| `obs[3:6]` | 3 | IMU Gyro | Torso angular velocity $[\omega_x, \omega_y, \omega_z]$ from STM32 `board.get_imu()` |
| `obs[6:9]` | 3 | IMU Accelerometer | Torso linear acceleration $[a_x, a_y, a_z]$ from STM32 `board.get_imu()` |
| `obs[9:27]` | 18 | Joint Pos Offsets | $q_i - q_{\text{default}}$ from LX-16A bus servos (`ctl.get_bus_servo_pulse`) |
| `obs[27:45]` | 18 | Joint Velocities | $\dot{q}_i$ ($rad/s$) via numerical differentiation of servo pulse feedback |
| `obs[45:63]` | 18 | Previous Actions | $a_{t-1} \in [-1, 1]^{18}$ (internal state history) |

*Total Core Dimensions: 63*.

### Optional Simulation-Only Additions:
- `include_lin_vel=True`: Appends 3 dims `body_lin_vel` ($[v_x, v_y, v_z]$). **Note**: Real robots cannot measure this directly; keep `False` for real-world deployment.
- `include_foot_contacts=True`: Appends 2 dims (`l_foot_touch`, `r_foot_touch`). Real TonyPi feet have no pressure sensors.

---

## 5. Reward Function Breakdown

The reward is computed in `_compute_reward(action)`:

1. **Forward Velocity**: $+w_{\text{fwd}} \cdot v_x$ (encourages moving forward).
2. **Alive Bonus**: $+w_{\text{alive}}$ (constant reward each step robot does not fall).
3. **Lateral Penalty**: $-w_{\text{lat}} \cdot v_y^2$ (penalizes drifting sideways).
4. **Yaw Penalty**: $-w_{\text{yaw}} \cdot \omega_z^2$ (penalizes spinning/turning).
5. **Height Penalty**: $-w_{\text{height}} \cdot (z - z_{\text{target}})^2$ (maintains standing height $\approx 0.262\,\text{m}$).
6. **Upright Penalty**: $-w_{\text{upright}} \cdot \max(0, 1 - \text{up}_z)$ (penalizes tilting).
7. **Control Effort**: $-w_{\text{ctrl}} \cdot \|a\|^2$ (energy efficiency).
8. **Smoothness**: $-w_{\text{smooth}} \cdot \|a_t - a_{t-1}\|^2$ (penalizes jerky actions).

---

## 6. Sim-to-Real Hardware Deployment

To transfer a policy trained in this environment to the real physical TonyPi:

1. **Policy Input Mapping**:
   * Feed real MPU-6050 accelerometer and gyroscope readings to `obs[0:9]`.
   * Feed real LX-16A servo positions and differences from `getBusServoPulse(id)` to `obs[9:27]`.
2. **Policy Output Mapping**:
   * Take the policy's normalized output $a_i \in [-1, 1]$.
   * Convert to angle in radians via `action_to_joint_targets(a)`.
   * Convert angle in radians to Hiwonder servo pulses (0–1000):
     $$\text{pulse} = \text{round}\left(500 + \frac{\theta_{\text{deg}}}{240.0} \times 1000\right)$$
   * Send via `Board.setBusServoPulse(servo_id, pulse, use_time)`.
