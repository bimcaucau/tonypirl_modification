# TonyPi Gymnasium Environment
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class TonyPiEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    JOINT_NAMES = [
        "neck_yaw",           # 0
        "head_pitch",         # 1
        "l_shoulder_pitch",   # 2
        "l_upperarm_roll",    # 3
        "l_elbow",            # 4
        "r_shoulder_pitch",   # 5
        "r_upperarm_roll",    # 6
        "r_elbow",            # 7
        "l_hip_roll",         # 8
        "l_hip_pitch",        # 9
        "l_knee",             # 10
        "l_ankle_pitch",      # 11
        "l_ankle_roll",       # 12
        "r_hip_roll",         # 13
        "r_hip_pitch",        # 14
        "r_knee",             # 15
        "r_ankle_pitch",      # 16
        "r_ankle_roll",       # 17
    ]

    def __init__(
        self,
        xml_path: str = None,
        frame_skip: int = 10,
        action_mode: str = "pi_scaled_offset",
        action_scale: float = 0.25,
        render_mode: str = None,
        target_forward_velocity: float = 0.20,
        vel_tracking_sigma: float = 0.08,
        overspeed_penalty_weight: float = 2.0,
        forward_vel_weight: float = 3.0,
        alive_weight: float = 1.0,
        vertical_vel_weight: float = 2.5,
        feet_air_penalty: float = 3.5,
        foot_slip_weight: float = 1.0,
        prolonged_stance_penalty: float = 0.20,
        straight_line_pos_weight: float = 2.0,
        lateral_vel_weight: float = 1.5,
        heading_yaw_weight: float = 1.5,
        angular_vel_weight: float = 0.5,
        foot_switch_reward: float = 3.0,
        single_support_reward: float = 0.8,
        foot_clearance_reward: float = 0.8,
        min_stance_steps: int = 5,
        max_stance_steps: int = 20,
        height_weight: float = 3.0,
        upright_weight: float = 2.5,
        arm_posture_weight: float = 0.5,
        ctrl_cost_weight: float = 0.005,
        smooth_cost_weight: float = 0.03,
        target_height: float = 0.262,
        reset_noise_scale: float = 0.02,
        include_lin_vel: bool = False,
        include_foot_contacts: bool = False,
    ):
        super().__init__()
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "tonypi_mujoco", "scene.xml")
        elif not os.path.isabs(xml_path):
            if not os.path.exists(xml_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                candidate = os.path.join(base_dir, xml_path)
                if os.path.exists(candidate):
                    xml_path = candidate

        xml_path = os.path.abspath(xml_path)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.action_mode = action_mode
        self.action_scale = action_scale
        self.render_mode = render_mode
        self.include_lin_vel = include_lin_vel
        self.include_foot_contacts = include_foot_contacts

        self.target_forward_velocity = target_forward_velocity
        self.vel_tracking_sigma = vel_tracking_sigma
        self.overspeed_penalty_weight = overspeed_penalty_weight
        self.forward_vel_weight = forward_vel_weight
        self.alive_weight = alive_weight
        self.vertical_vel_weight = vertical_vel_weight
        self.feet_air_penalty = feet_air_penalty
        self.foot_slip_weight = foot_slip_weight
        self.prolonged_stance_penalty = prolonged_stance_penalty
        self.straight_line_pos_weight = straight_line_pos_weight
        self.lateral_vel_weight = lateral_vel_weight
        self.heading_yaw_weight = heading_yaw_weight
        self.angular_vel_weight = angular_vel_weight
        self.foot_switch_reward = foot_switch_reward
        self.single_support_reward = single_support_reward
        self.foot_clearance_reward = foot_clearance_reward
        self.min_stance_steps = min_stance_steps
        self.max_stance_steps = max_stance_steps
        self.height_weight = height_weight
        self.upright_weight = upright_weight
        self.arm_posture_weight = arm_posture_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.smooth_cost_weight = smooth_cost_weight
        self.target_height = target_height
        self.reset_noise_scale = reset_noise_scale

        self.num_joints = len(self.JOINT_NAMES)
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.l_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_foot")
        self.r_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_foot")
        self.l_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_site")
        self.r_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_site")

        self.joint_qpos_adr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.JOINT_NAMES
        ], dtype=np.int32)

        self.joint_qvel_adr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.JOINT_NAMES
        ], dtype=np.int32)

        # Precompute arm joint indices for upper body posture stabilization
        self.ARM_JOINT_INDICES = np.array([2, 3, 4, 5, 6, 7], dtype=np.int32)
        self.arm_joint_qpos_adr = self.joint_qpos_adr[self.ARM_JOINT_INDICES]

        self.joint_limits_lower = np.array([
            self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name), 0]
            for name in self.JOINT_NAMES
        ], dtype=np.float32)
        self.joint_limits_upper = np.array([
            self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name), 1]
            for name in self.JOINT_NAMES
        ], dtype=np.float32)

        self.default_standing_qpos = np.zeros(self.num_joints, dtype=np.float32)
        self.default_root_pos = np.array([0.0, 0.0, 0.267], dtype=np.float32)
        self.default_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # Core observation (real hardware signals only: 63 dims):
        # 3: projected gravity (IMU orientation)
        # 3: IMU gyro (angular velocity rad/s)
        # 3: IMU accelerometer (linear acceleration m/s^2)
        # 18: joint position offsets (rad)
        # 18: joint velocities (rad/s)
        # 18: previous actions [-1, 1]
        obs_dim = 3 + 3 + 3 + 18 + 18 + 18
        if self.include_lin_vel:
            obs_dim += 3  # +3 body_lin_vel (sim-only, not on real robot)
        if self.include_foot_contacts:
            obs_dim += 2  # +2 foot touch sensors (sim-only, not on real robot)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self.step_count = 0
        self.viewer = None
        self.last_stance_foot = 0
        self.stance_step_count = 0
        self.total_foot_switches = 0
        self._c_force_buf = np.zeros(6, dtype=np.float64)

    def action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        if self.action_mode == "pi_scaled_offset":
            scale = self.action_scale * np.pi
            targets = self.default_standing_qpos + action * scale
        elif self.action_mode == "pi_direct":
            scale = self.action_scale * np.pi
            targets = action * scale
        elif self.action_mode == "joint_limits":
            targets = self.joint_limits_lower + 0.5 * (action + 1.0) * (
                self.joint_limits_upper - self.joint_limits_lower
            )
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        targets = np.clip(targets, self.joint_limits_lower, self.joint_limits_upper)
        return targets

    def _get_obs(self) -> np.ndarray:
        rot_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        projected_gravity = -rot_mat[2, :]
        imu_gyro = self.data.sensor("imu_gyro").data.copy()
        imu_accel = self.data.sensor("imu_accel").data.copy()
        current_joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_pos_diff = (current_joint_pos - self.default_standing_qpos).astype(np.float32)
        joint_vel = self.data.qvel[self.joint_qvel_adr].astype(np.float32)
        # world_lin_vel = self.data.qvel[0:3]
        # body_lin_vel = rot_mat.T @ world_lin_vel

        obs_components = [
            projected_gravity,
            imu_gyro,
            imu_accel,
            joint_pos_diff,
            joint_vel,
            self.prev_action,
        ]
        if self.include_lin_vel:
            world_lin_vel = self.data.qvel[0:3]
            body_lin_vel = rot_mat.T @ world_lin_vel
            obs_components.append(body_lin_vel)
        if self.include_foot_contacts:
            l_touch = self.data.sensor("l_foot_touch").data.copy()
            r_touch = self.data.sensor("r_foot_touch").data.copy()
            obs_components.extend([l_touch, r_touch])

        return np.concatenate(obs_components).astype(np.float32)

    def _get_foot_forces(self) -> tuple[float, float]:
        """Compute normal contact forces on left and right feet from contacts (simulation only)."""
        l_force = 0.0
        r_force = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            mujoco.mj_contactForce(self.model, self.data, i, self._c_force_buf)
            fn = max(0.0, self._c_force_buf[0])
            if b1 == self.l_foot_body_id or b2 == self.l_foot_body_id:
                l_force += fn
            if b1 == self.r_foot_body_id or b2 == self.r_foot_body_id:
                r_force += fn
        return float(l_force), float(r_force)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        # 1. Velocities & Poses
        vx, vy, vz = self.data.qvel[0:3]
        wx, wy, wz = self.data.qvel[3:6]
        pelvis_pos = self.data.xpos[self.pelvis_body_id]
        pelvis_y = pelvis_pos[1]
        pelvis_z = pelvis_pos[2]
        rot_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        up_z = rot_mat[2, 2]

        # 2. Straight-Line Heading Alignment (+X axis)
        heading_x = rot_mat[0, 0]
        heading_y = rot_mat[1, 0]
        yaw_heading_error = np.arctan2(heading_y, heading_x)

        # 3. Foot Contacts & Positions
        l_force, r_force = self._get_foot_forces()
        is_left_contact = (l_force > 2.0)
        is_right_contact = (r_force > 2.0)

        l_foot_x = self.data.site_xpos[self.l_foot_site_id][0]
        r_foot_x = self.data.site_xpos[self.r_foot_site_id][0]
        l_foot_z = self.data.site_xpos[self.l_foot_site_id][2]
        r_foot_z = self.data.site_xpos[self.r_foot_site_id][2]

        # 4. Anti-Hopping: Flight Phase Penalty & Single Support
        r_single_support = 0.0
        p_air = 0.0
        if is_left_contact and not is_right_contact:
            current_stance = 1  # Left stance, Right swing
            r_single_support = self.single_support_reward
        elif is_right_contact and not is_left_contact:
            current_stance = 2  # Right stance, Left swing
            r_single_support = self.single_support_reward
        elif is_left_contact and is_right_contact:
            current_stance = 3  # Double support (transfer)
        else:
            current_stance = 0  # Both in air (hopping / airborne flight)
            p_air = -self.feet_air_penalty

        # 5. Stance Switching & Anti-Hopping Prolonged Stance Penalty
        r_foot_switch = 0.0
        p_prolonged = 0.0
        if current_stance in (1, 2):
            if current_stance == self.last_stance_foot:
                self.stance_step_count += 1
                if self.stance_step_count > self.max_stance_steps:
                    # Penalize hopping on one leg or dragging without stepping
                    p_prolonged = -self.prolonged_stance_penalty * (self.stance_step_count - self.max_stance_steps)
            else:
                # Successfully alternated stance foot (Left <-> Right)
                if self.last_stance_foot in (1, 2) and self.stance_step_count >= self.min_stance_steps:
                    # Reward taking an actual forward step (new stance foot planted in front)
                    step_advancement = (l_foot_x - r_foot_x) if current_stance == 1 else (r_foot_x - l_foot_x)
                    fwd_step_bonus = 1.0 + np.clip(step_advancement / 0.04, 0.0, 1.5)
                    r_foot_switch = self.foot_switch_reward * fwd_step_bonus
                    self.total_foot_switches += 1

                self.last_stance_foot = current_stance
                self.stance_step_count = 1

        # 6. Swing Foot Clearance Reward (lifts foot 1.5 - 3.5 cm to avoid dragging)
        r_clearance = 0.0
        if current_stance == 1:  # Left stance -> reward right foot lift
            lift = np.clip(r_foot_z - 0.005, 0.0, 0.035)
            r_clearance = self.foot_clearance_reward * (lift / 0.035)
        elif current_stance == 2:  # Right stance -> reward left foot lift
            lift = np.clip(l_foot_z - 0.005, 0.0, 0.035)
            r_clearance = self.foot_clearance_reward * (lift / 0.035)

        # 7. Foot Slip Penalty (when foot is in contact, its horizontal speed should be ~0)
        l_slip = 0.0
        r_slip = 0.0
        if is_left_contact:
            l_slip = float(self.data.cvel[self.l_foot_body_id, 3] ** 2 + self.data.cvel[self.l_foot_body_id, 4] ** 2)
        if is_right_contact:
            r_slip = float(self.data.cvel[self.r_foot_body_id, 3] ** 2 + self.data.cvel[self.r_foot_body_id, 4] ** 2)
        p_slip = -self.foot_slip_weight * (l_slip + r_slip)

        # 8. Anti-Hopping: Vertical Bouncing Penalty
        p_vz = -self.vertical_vel_weight * (vz ** 2)

        # 9. Velocity Tracking (Gaussian bell curve centered at target speed)
        # Replaces linear vx reward to prevent rushing, launching, or hopping forward
        vel_error = vx - self.target_forward_velocity
        r_forward = self.forward_vel_weight * float(np.exp(- (vel_error ** 2) / (self.vel_tracking_sigma ** 2)))
        overspeed = max(0.0, vx - (self.target_forward_velocity + 0.10))
        p_overspeed = -self.overspeed_penalty_weight * (overspeed ** 2)

        r_alive = self.alive_weight

        # 10. Heading, Lateral, and Torso Angular Velocity Penalties
        p_straight_line = -self.straight_line_pos_weight * (pelvis_y ** 2)
        p_lateral = -self.lateral_vel_weight * (vy ** 2)
        p_yaw_heading = -self.heading_yaw_weight * (yaw_heading_error ** 2)
        p_ang_vel = -self.angular_vel_weight * (wx ** 2 + wy ** 2 + 0.5 * wz ** 2)

        p_height = -self.height_weight * ((pelvis_z - self.target_height) ** 2)
        p_upright = -self.upright_weight * max(0.0, 1.0 - up_z)

        # 11. Arm Posture Stabilization (keeps arms quiet instead of flailing for jumping momentum)
        arm_diff = self.data.qpos[self.arm_joint_qpos_adr] - self.default_standing_qpos[self.ARM_JOINT_INDICES]
        p_arm = -self.arm_posture_weight * float(np.sum(np.square(arm_diff)))

        p_ctrl = -self.ctrl_cost_weight * np.sum(np.square(action))
        p_smooth = -self.smooth_cost_weight * np.sum(np.square(action - self.prev_action))

        total_reward = (
            r_forward + p_overspeed + r_alive +
            p_straight_line + p_lateral + p_yaw_heading + p_ang_vel +
            r_foot_switch + r_single_support + r_clearance +
            p_air + p_vz + p_slip + p_prolonged +
            p_height + p_upright + p_arm + p_ctrl + p_smooth
        )

        info = {
            "reward_forward": float(r_forward),
            "reward_alive": float(r_alive),
            "reward_foot_switch": float(r_foot_switch),
            "reward_single_support": float(r_single_support),
            "reward_clearance": float(r_clearance),
            "penalty_overspeed": float(p_overspeed),
            "penalty_air": float(p_air),
            "penalty_vz": float(p_vz),
            "penalty_slip": float(p_slip),
            "penalty_prolonged": float(p_prolonged),
            "penalty_straight_line": float(p_straight_line),
            "penalty_yaw_heading": float(p_yaw_heading),
            "penalty_lateral": float(p_lateral),
            "penalty_ang_vel": float(p_ang_vel),
            "penalty_height": float(p_height),
            "penalty_upright": float(p_upright),
            "penalty_arm": float(p_arm),
            "penalty_ctrl": float(p_ctrl),
            "penalty_smooth": float(p_smooth),
            "forward_vel": float(vx),
            "vertical_vel": float(vz),
            "pelvis_y": float(pelvis_y),
            "pelvis_z": float(pelvis_z),
            "total_switches": int(self.total_foot_switches),
            "l_force": float(l_force),
            "r_force": float(r_force),
        }
        return float(total_reward), info

    def _is_terminated(self) -> bool:
        pelvis_z = self.data.qpos[2]
        # Prevent airborne hopping (>0.32m) or collapsing (<0.18m)
        if pelvis_z < 0.18 or pelvis_z > 0.32:
            return True
        rot_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        up_z = rot_mat[2, 2]
        if up_z < 0.65:
            return True
        return False

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        joint_targets = self.action_to_joint_targets(action)
        self.data.ctrl[:] = joint_targets

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = False

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, reward_info

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = self.default_root_pos
        self.data.qpos[3:7] = self.default_root_quat

        noise = self.np_random.uniform(
            -self.reset_noise_scale, self.reset_noise_scale, size=self.num_joints
        )
        perturbed_joints = self.default_standing_qpos + noise
        perturbed_joints = np.clip(
            perturbed_joints, self.joint_limits_lower, self.joint_limits_upper
        )

        self.data.qpos[self.joint_qpos_adr] = perturbed_joints
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_standing_qpos

        mujoco.mj_forward(self.model, self.data)

        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self.step_count = 0
        self.last_stance_foot = 0
        self.stance_step_count = 0
        self.total_foot_switches = 0

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera="tracking")
            return renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_joint_positions(self) -> dict[str, float]:
        return {
            name: float(self.data.qpos[adr])
            for name, adr in zip(self.JOINT_NAMES, self.joint_qpos_adr)
        }

    def get_joint_velocities(self) -> dict[str, float]:
        return {
            name: float(self.data.qvel[adr])
            for name, adr in zip(self.JOINT_NAMES, self.joint_qvel_adr)
        }

    def get_sensor_readings(self) -> dict[str, np.ndarray]:
        readings = {}
        for i in range(self.model.nsensor):
            sname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            readings[sname] = self.data.sensor(sname).data.copy()
        return readings

    def get_pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.pelvis_body_id].copy()
        rot = self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()
        return pos, rot
