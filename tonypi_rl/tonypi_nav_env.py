"""High-level TonyPi navigation environment.

Policy outputs a discrete locomotion command (same names as ActionGroupDict).
In sim the floating base is moved kinematically while joints stay in stand pose.
Observations: goal + 1 ToF (VL53L0X) + 1 sonar + IMU + last action, plus a
64x64 occupancy mask from MuJoCo ground-truth head-camera segmentation.
Vec is polar 18-D: dist/3.0, sin(heading), cos(heading), ToF, sonar, gravity, gyro, last action.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from tonypi_rl.perception import MujocoGtOccupancy, OCCUPANCY_SIZE

ACTION_NAMES = [
    "stand",
    "go_forward",
    "back_fast",
    "left_move_fast",
    "right_move_fast",
    "turn_left",
    "turn_right",
]

OPPOSITE_NAME = {
    "go_forward": "back_fast",
    "back_fast": "go_forward",
    "left_move_fast": "right_move_fast",
    "right_move_fast": "left_move_fast",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}

GOAL_DIST_SCALE = 3.0

# Body-frame primitive (dx_m, dy_m, dyaw_rad) matching one real gait cycle.
PRIMITIVES = {
    "stand": (0.00, 0.00, 0.00),
    "go_forward": (0.08, 0.00, 0.00),
    "back_fast": (-0.06, 0.00, 0.00),
    "left_move_fast": (0.00, 0.05, 0.00),
    "right_move_fast": (0.00, -0.05, 0.00),
    "turn_left": (0.00, 0.00, 0.35),
    "turn_right": (0.00, 0.00, -0.35),
}

TOF_NAMES = ["tof_0"]  # stock TonyPi has a single VL53L0X; camera occupancy covers sides
TOF_CUTOFF = 1.2
SONAR_CUTOFF = 5.0
N_OBS_SLOTS = 8
HIDDEN_XY = (12.0, 12.0)
STAND_Z = 0.267
OBS_Z = 0.22

GAP_MERGED = 0.16      # center-to-center: treat as one blob
GAP_PASSABLE = 0.55    # center-to-center: robot can walk between
START_CLEAR = 0.55     # min hypot(box, origin)
GOAL_CLEAR = 0.45

# demo layout: goal at (+2.0, -0.5) with a few boxes to weave around
DEMO_GOAL = (2.0, -0.5)
DEMO_OBSTACLES = [
    (0.95, 0.02, OBS_Z),
    (1.35, -0.48, OBS_Z),
    (1.60, 0.42, OBS_Z),
    (0.72, -0.36, OBS_Z),
]


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float64)


class TonyPiNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    JOINT_NAMES = [
        "neck_yaw", "head_pitch",
        "l_shoulder_pitch", "l_upperarm_roll", "l_elbow",
        "r_shoulder_pitch", "r_upperarm_roll", "r_elbow",
        "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
        "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
    ]

    def __init__(
        self,
        xml_path: str = None,
        render_mode: str = None,
        goal_mode: str = "fixed",
        layout: str = "demo",
        random_spawn: bool = False,
        random_robot_yaw: bool = False,
        n_obstacles: int = 4,
        max_episode_steps: int = 160,
        primitive_substeps: int = 16,
        arrive_radius: float = 0.20,
        progress_weight: float = 5.0,
        arrive_bonus: float = 100.0,
        time_penalty: float = 0.25,
        collision_penalty: float = 3.0,
        reverse_penalty: float = 0.50,
        oscillation_penalty: float = 0.30,
        heading_align_weight: float = 0.15,
        ram_penalty: float = 1.0,
        collision_streak_limit: int = 8,
        use_geodesic_reward: bool = False,
        expert_robot_radius: float = 0.18,
        expert_grid_res: float = 0.05,
        expert_heading_weight: float = 0.12,
        geodesic_progress_clip: float = 0.15,
    ):
        super().__init__()
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "tonypi_mujoco", "scene_nav.xml")
        elif not os.path.isabs(xml_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base_dir, xml_path)
            if os.path.exists(candidate):
                xml_path = candidate
        xml_path = os.path.abspath(xml_path)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Nav scene XML not found: {xml_path}")

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        if random_spawn:
            if layout in ("demo", "fixed"):
                layout = "mixed"
            goal_mode = "random"
        self.random_spawn = bool(random_spawn)
        self.random_robot_yaw = bool(random_robot_yaw)
        self.goal_mode = goal_mode
        self.layout = layout
        self._layout_kind = layout
        self.n_obstacles = int(n_obstacles)
        self.max_episode_steps = int(max_episode_steps)
        self.primitive_substeps = int(primitive_substeps)
        self.arrive_radius = float(arrive_radius)
        self.progress_weight = float(progress_weight)
        self.arrive_bonus = float(arrive_bonus)
        self.time_penalty = float(time_penalty)
        self.collision_penalty = float(collision_penalty)
        self.reverse_penalty = float(reverse_penalty)
        self.oscillation_penalty = float(oscillation_penalty)
        self.heading_align_weight = float(heading_align_weight)
        self.ram_penalty = float(ram_penalty)
        self.collision_streak_limit = int(collision_streak_limit)
        self.use_geodesic_reward = bool(use_geodesic_reward)
        self.expert_robot_radius = float(expert_robot_radius)
        self.expert_grid_res = float(expert_grid_res)
        self.expert_heading_weight = float(expert_heading_weight)
        self.geodesic_progress_clip = float(geodesic_progress_clip)
        self._expert = None
        self._prev_progress = 0.0
        self._collision_streak = 0

        self.num_joints = len(self.JOINT_NAMES)
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.joint_qpos_adr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.JOINT_NAMES
        ], dtype=np.int32)
        self.default_standing_qpos = np.zeros(self.num_joints, dtype=np.float64)

        self.obs_body_ids = []
        self.obs_geom_ids = []
        self.obs_mocap_ids = []
        for i in range(N_OBS_SLOTS):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            self.obs_body_ids.append(bid)
            self.obs_geom_ids.append(gid)
            self.obs_mocap_ids.append(int(self.model.body_mocapid[bid]))
        self.obs_geom_ids_arr = np.array(self.obs_geom_ids, dtype=np.int32)

        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body_id])

        self.perception = MujocoGtOccupancy(
            self.model, camera="head_camera", obstacle_geom_ids=self.obs_geom_ids_arr,
            use_gl=False,
        )

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.vec_dim = 3 + len(TOF_NAMES) + 1 + 3 + 3 + len(ACTION_NAMES)
        self.observation_space = spaces.Dict({
            "vec": spaces.Box(-np.inf, np.inf, shape=(self.vec_dim,), dtype=np.float32),
            "occupancy": spaces.Box(0.0, 1.0, shape=(1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32),
        })

        self.viewer = None
        self.track_renderer = None
        self.goal_xy = np.array(DEMO_GOAL, dtype=np.float64)
        self.prev_action = 0
        self.step_count = 0
        self._prev_dist = 0.0
        self._active_obs = []

    def _read_range(self, name: str, cutoff: float) -> float:
        d = float(self.data.sensor(name).data[0])
        if not np.isfinite(d) or d < 0.0:
            d = cutoff
        return float(np.clip(d, 0.0, cutoff))

    def _pelvis_yaw(self) -> float:
        return _yaw_from_quat(self.data.qpos[3:7])

    def _goal_body_frame(self) -> tuple[float, float, float, float]:
        px, py = float(self.data.qpos[0]), float(self.data.qpos[1])
        dx = float(self.goal_xy[0] - px)
        dy = float(self.goal_xy[1] - py)
        yaw = self._pelvis_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        gx = c * dx + s * dy
        gy = -s * dx + c * dy
        dist = float(np.hypot(dx, dy))
        heading = float(np.arctan2(gy, gx))
        return gx, gy, dist, heading

    def _set_root(self, x: float, y: float, yaw: float):
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = STAND_Z
        self.data.qpos[3:7] = _quat_from_yaw(yaw)
        self.data.qpos[self.joint_qpos_adr] = self.default_standing_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_standing_qpos
        mujoco.mj_forward(self.model, self.data)

    def _colliding(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.obs_geom_ids_arr or c.geom2 in self.obs_geom_ids_arr:
                return True
        return False

    def _hide_all_obstacles(self):
        for mid in self.obs_mocap_ids:
            self.data.mocap_pos[mid] = [HIDDEN_XY[0] + mid, HIDDEN_XY[1], OBS_Z]
            self.data.mocap_quat[mid] = [1, 0, 0, 0]

    def _place_goal(self, xy):
        self.goal_xy = np.array(xy, dtype=np.float64)
        self.data.mocap_pos[self.goal_mocap_id] = [self.goal_xy[0], self.goal_xy[1], 0.0]
        self.data.mocap_quat[self.goal_mocap_id] = [1, 0, 0, 0]

    def _sample_goal(self) -> np.ndarray:
        if self.goal_mode == "random":
            ang = float(self.np_random.uniform(-np.pi, np.pi))
            rad = float(self.np_random.uniform(1.0, 2.5))
            goal = np.array([rad * np.cos(ang), rad * np.sin(ang)], dtype=np.float64)
        else:
            goal = np.array(DEMO_GOAL, dtype=np.float64)
        self._place_goal(goal)
        return goal

    def _choose_layout_kind(self) -> str:
        if self.layout in ("mixed", "curriculum"):
            kinds = ["scatter", "wall", "u_trap", "gate"]
            probs = [0.40, 0.30, 0.15, 0.15]
            return str(self.np_random.choice(kinds, p=probs))
        if self.layout in ("random", "scatter"):
            return "scatter"
        if self.layout in ("wall", "u_trap", "gate", "empty", "demo"):
            return self.layout
        return "scatter"

    def _place_scatter(self, goal: np.ndarray):
        self._hide_all_obstacles()
        n = int(self.np_random.integers(max(1, self.n_obstacles - 1), self.n_obstacles + 1))
        n = min(n, N_OBS_SLOTS)
        placed = []
        slot_idx = 0

        # With prob 0.5, first place a merged pair on start-goal segment (LOS block)
        if float(self.np_random.uniform(0.0, 1.0)) < 0.5 and n >= 2:
            goal_dist = float(np.hypot(goal[0], goal[1]))
            u = goal / max(1e-6, goal_dist)
            t = np.array([-u[1], u[0]])
            m_dist = float(np.clip(0.5 * goal_dist, START_CLEAR + 0.05, goal_dist - GOAL_CLEAR - 0.05))
            m = m_dist * u
            b1 = m + 0.0795 * t
            b2 = m - 0.0795 * t
            if (np.hypot(b1[0], b1[1]) >= START_CLEAR and
                np.hypot(b2[0], b2[1]) >= START_CLEAR and
                np.hypot(b1[0] - goal[0], b1[1] - goal[1]) >= GOAL_CLEAR and
                np.hypot(b2[0] - goal[0], b2[1] - goal[1]) >= GOAL_CLEAR):
                for b in (b1, b2):
                    z = OBS_Z
                    self.data.mocap_pos[self.obs_mocap_ids[slot_idx]] = [b[0], b[1], z]
                    placed.append((float(b[0]), float(b[1]), z))
                    slot_idx += 1

        for _ in range(slot_idx, n):
            for _ in range(50):
                r = float(self.np_random.uniform(0.55, 2.3))
                a = float(self.np_random.uniform(-np.pi, np.pi))
                x, y = r * np.cos(a), r * np.sin(a)
                if np.hypot(x, y) < START_CLEAR:
                    continue
                if np.hypot(x - goal[0], y - goal[1]) < GOAL_CLEAR:
                    continue
                conflict = False
                for px, py, _ in placed:
                    d = float(np.hypot(x - px, y - py))
                    if GAP_MERGED < d < GAP_PASSABLE:
                        conflict = True
                        break
                if conflict:
                    continue
                z = OBS_Z
                self.data.mocap_pos[self.obs_mocap_ids[slot_idx]] = [x, y, z]
                placed.append((x, y, z))
                slot_idx += 1
                break
        self._active_obs = placed

    def _place_wall(self, goal: np.ndarray):
        self._hide_all_obstacles()
        goal_dist = float(np.hypot(goal[0], goal[1]))
        u = goal / max(1e-6, goal_dist)
        t = np.array([-u[1], u[0]])

        m_dist = float(np.clip(0.5 * goal_dist, START_CLEAR + 0.05, goal_dist - GOAL_CLEAR - 0.05))
        m = m_dist * u

        n_wall = int(self.np_random.integers(4, 6))
        n_wall = min(n_wall, N_OBS_SLOTS)
        s = 0.15

        side = float(self.np_random.choice([-1.0, 1.0]))
        if n_wall == 4:
            offsets = [-1, 0, 1, 2] if side > 0 else [-2, -1, 0, 1]
        else:
            offsets = [-1, 0, 1, 2, 3] if side > 0 else [-3, -2, -1, 0, 1]

        placed = []
        for i, k in enumerate(offsets):
            pos_xy = m + (k * s) * t
            x, y = float(pos_xy[0]), float(pos_xy[1])
            z = OBS_Z
            self.data.mocap_pos[self.obs_mocap_ids[i]] = [x, y, z]
            placed.append((x, y, z))
        self._active_obs = placed

    def _place_u_trap(self, goal: np.ndarray):
        self._hide_all_obstacles()
        goal_dist = float(np.hypot(goal[0], goal[1]))
        u = goal / max(1e-6, goal_dist)
        t = np.array([-u[1], u[0]])

        d_back = float(self.np_random.uniform(0.55, 0.70))
        d_back = min(d_back, goal_dist - GOAL_CLEAR - 0.05)
        d_back = max(0.50, d_back)

        # 3 merged boxes for back wall perpendicular to u (in front of robot)
        boxes = [
            d_back * u,
            d_back * u - 0.15 * t,
            d_back * u + 0.15 * t,
        ]
        # Legs along u at ± w_leg * t (w_leg = 0.28)
        # One pair near back wall (d_back - 0.15), one pair behind origin (-0.25)
        w_leg = 0.28
        boxes.append((d_back - 0.15) * u - w_leg * t)
        boxes.append(-0.25 * u - w_leg * t)
        boxes.append((d_back - 0.15) * u + w_leg * t)
        boxes.append(-0.25 * u + w_leg * t)

        placed = []
        for i, b in enumerate(boxes[:N_OBS_SLOTS]):
            x, y = float(b[0]), float(b[1])
            z = OBS_Z
            self.data.mocap_pos[self.obs_mocap_ids[i]] = [x, y, z]
            placed.append((x, y, z))
        self._active_obs = placed

    def _place_gate(self, goal: np.ndarray):
        self._hide_all_obstacles()
        goal_dist = float(np.hypot(goal[0], goal[1]))
        u = goal / max(1e-6, goal_dist)
        t = np.array([-u[1], u[0]])

        d_gate = float(np.clip(0.5 * goal_dist, START_CLEAR + 0.05, goal_dist - GOAL_CLEAR - 0.05))
        c = d_gate * u
        side = float(self.np_random.choice([-1.0, 1.0]))
        offset_mag = float(self.np_random.uniform(0.35, 0.55))
        gap_center = c + (side * offset_mag) * t
        gap = 0.58

        boxes = [
            gap_center - 0.5 * gap * t,
            gap_center + 0.5 * gap * t,
        ]
        placed = []
        for i, b in enumerate(boxes):
            x, y = float(b[0]), float(b[1])
            z = OBS_Z
            self.data.mocap_pos[self.obs_mocap_ids[i]] = [x, y, z]
            placed.append((x, y, z))
        self._active_obs = placed

    def _sample_obstacles(self, goal: np.ndarray):
        self._hide_all_obstacles()
        kind = self._choose_layout_kind()
        self._layout_kind = kind
        if kind == "empty":
            self._active_obs = []
        elif kind == "demo":
            used = DEMO_OBSTACLES[: max(1, min(self.n_obstacles, len(DEMO_OBSTACLES)))]
            placed = []
            for i, (x, y, z) in enumerate(used):
                if np.hypot(x - goal[0], y - goal[1]) < GOAL_CLEAR:
                    continue
                self.data.mocap_pos[self.obs_mocap_ids[i]] = [x, y, z]
                placed.append((x, y, z))
            self._active_obs = placed
        elif kind == "wall":
            self._place_wall(goal)
        elif kind == "u_trap":
            self._place_u_trap(goal)
        elif kind == "gate":
            self._place_gate(goal)
        else:
            self._place_scatter(goal)

    def _sample_layout(self):
        for attempt in range(20):
            goal = self._sample_goal()
            self._sample_obstacles(goal)
            if self.use_geodesic_reward or self._expert is not None:
                self._rebuild_expert()
                if not self._expert.unreachable:
                    break
            else:
                break
        else:
            self._hide_all_obstacles()
            self._layout_kind = "scatter"
            self._place_scatter(self.goal_xy)
            if self.use_geodesic_reward or self._expert is not None:
                self._rebuild_expert()

    def _obstacle_rects(self):
        """Active boxes as (cx, cy, hx, hy). Parked slots (x>8) skipped."""
        rects = []
        for i, mid in enumerate(self.obs_mocap_ids):
            x, y, _z = self.data.mocap_pos[mid]
            if float(x) > 8.0:
                continue
            hx, hy, _hz = self.model.geom_size[self.obs_geom_ids[i]]
            rects.append((float(x), float(y), float(hx), float(hy)))
        return rects

    def _rebuild_expert(self):
        from tonypi_rl.nav_expert import PrimitiveExpert

        if self._expert is None:
            self._expert = PrimitiveExpert(
                ACTION_NAMES,
                PRIMITIVES,
                robot_radius=self.expert_robot_radius,
                grid_res=self.expert_grid_res,
                heading_weight=self.expert_heading_weight,
                arrive_radius=self.arrive_radius,
            )
        px, py = float(self.data.qpos[0]), float(self.data.qpos[1])
        self._expert.rebuild(
            self._obstacle_rects(),
            self.goal_xy,
            clear_xy=((px, py), tuple(self.goal_xy)),
        )

    def expert_action(self) -> int:
        if self._expert is None:
            self._rebuild_expert()
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        return int(self._expert.act(x, y, self._pelvis_yaw()))

    def geodesic_distance(self) -> float:
        if self._expert is None:
            self._rebuild_expert()
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        g = self._expert.geodesic(x, y)
        if np.isfinite(g):
            return float(g)
        return float("inf")

    def expert_path(self):
        if self._expert is None:
            self._rebuild_expert()
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        return self._expert.path_polyline(x, y)

    def _progress_metric(self) -> float:
        if self.use_geodesic_reward:
            g = self.geodesic_distance()
            if np.isfinite(g):
                return float(g)
            # stuck / inflated-occupied: punish, do NOT switch to Euclidean
            return float(self._prev_progress + self.geodesic_progress_clip)
        _, _, dist, _ = self._goal_body_frame()
        return float(dist)

    def _get_obs(self) -> dict:
        gx, gy, dist, heading = self._goal_body_frame()
        goal_vec = np.array(
            [dist / GOAL_DIST_SCALE, np.sin(heading), np.cos(heading)],
            dtype=np.float32,
        )
        tof = np.array([self._read_range(n, TOF_CUTOFF) / TOF_CUTOFF for n in TOF_NAMES], dtype=np.float32)
        sonar = np.array([self._read_range("sonar", SONAR_CUTOFF) / SONAR_CUTOFF], dtype=np.float32)
        rot = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        gravity = (-rot[2, :]).astype(np.float32)
        gyro = self.data.sensor("imu_gyro").data.copy().astype(np.float32)
        prev = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        prev[int(self.prev_action)] = 1.0
        vec = np.concatenate([
            goal_vec,
            tof, sonar, gravity, gyro, prev,
        ]).astype(np.float32)
        occupancy = self.perception.encode(self.data)
        return {"vec": vec, "occupancy": occupancy}

    def _compute_reward(
        self, collided: bool, expert_a: int, action: int, prev_action: int
    ) -> tuple[float, dict]:
        gx, gy, dist, heading = self._goal_body_frame()
        cur = self._progress_metric()
        raw_progress = self._prev_progress - cur
        if self.use_geodesic_reward:
            progress = float(np.clip(raw_progress, -self.geodesic_progress_clip, self.geodesic_progress_clip))
        else:
            progress = float(raw_progress)

        reward = self.progress_weight * progress - self.time_penalty
        arrived = dist < self.arrive_radius
        if arrived:
            reward += self.arrive_bonus
        if collided:
            reward -= self.collision_penalty

        tof0 = float(self._read_range("tof_0", TOF_CUTOFF))
        name = ACTION_NAMES[int(action)]
        prev_name = ACTION_NAMES[int(prev_action)]
        if name == "back_fast" and dist > 1.5 * self.arrive_radius and tof0 > 0.50:
            reward -= self.reverse_penalty
        if OPPOSITE_NAME.get(name) == prev_name:
            reward -= self.oscillation_penalty
        if name == "go_forward" and tof0 > 0.50 and not collided:
            reward += self.heading_align_weight * float(np.cos(heading))
        if name == "go_forward" and tof0 < 0.35:
            reward -= self.ram_penalty

        geo = self.geodesic_distance() if self._expert is not None else float("inf")
        info = {
            "dist": float(dist),
            "progress": float(progress),
            "arrived": bool(arrived),
            "collision": bool(collided),
            "collision_streak": int(self._collision_streak),
            "gx": float(gx),
            "gy": float(gy),
            "heading": float(heading),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "tof": [self._read_range(n, TOF_CUTOFF) for n in TOF_NAMES],
            "sonar": self._read_range("sonar", SONAR_CUTOFF),
            "layout": str(self.layout),
            "layout_kind": str(self._layout_kind),
            "action_name": name,
            "expert_action": int(expert_a),
            "geodesic": float(geo) if np.isfinite(geo) else -1.0,
            "expert_unreachable": bool(self._expert is not None and self._expert.unreachable),
        }
        self._prev_dist = dist
        self._prev_progress = cur
        return float(reward), info

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)

        if options.get("random_spawn") or options.get("random"):
            if self.layout in ("demo", "fixed"):
                self.layout = "mixed"
            self.goal_mode = "random"
        if options.get("random_target"):
            self.goal_mode = "random"
        if options.get("random_obstacles"):
            if self.layout in ("demo", "fixed"):
                self.layout = "mixed"
        if "layout" in options:
            self.layout = options["layout"]
        if "goal_mode" in options:
            self.goal_mode = options["goal_mode"]

        if "goal" in options:
            self.goal_mode = "fixed"
            self._place_goal(options["goal"])
            if "obstacles" in options:
                self._hide_all_obstacles()
                self._active_obs = []
                for i, pos in enumerate(options["obstacles"][:N_OBS_SLOTS]):
                    self.data.mocap_pos[self.obs_mocap_ids[i]] = [pos[0], pos[1], pos[2] if len(pos) > 2 else OBS_Z]
                    self._active_obs.append(tuple(pos))
                self._layout_kind = "custom"
            else:
                self._sample_obstacles(self.goal_xy)
        else:
            self._sample_layout()

        if "yaw" in options:
            yaw0 = float(options["yaw"])
        elif options.get("random_yaw") or options.get("random_robot_yaw") or self.random_robot_yaw:
            yaw0 = float(self.np_random.uniform(-np.pi, np.pi))
        else:
            yaw0 = 0.0
        self._set_root(0.0, 0.0, yaw0)
        self.prev_action = 0
        self.step_count = 0
        self._collision_streak = 0
        # Stock PPO (no geodesic) skips the privileged grid. expert_action() rebuilds lazily.
        if self.use_geodesic_reward:
            self._rebuild_expert()
        else:
            self._expert = None
        gx, gy, self._prev_dist, heading = self._goal_body_frame()
        self._prev_progress = self._progress_metric()
        obs = self._get_obs()
        if self._expert is not None:
            geo = self.geodesic_distance()
            expert_a = int(self.expert_action())
            unreachable = bool(self._expert.unreachable)
        else:
            geo = float("inf")
            expert_a = -1
            unreachable = False
        info = {
            "dist": float(self._prev_dist),
            "gx": float(gx),
            "gy": float(gy),
            "heading": float(heading),
            "collision_streak": 0,
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "tof": [self._read_range(n, TOF_CUTOFF) for n in TOF_NAMES],
            "sonar": self._read_range("sonar", SONAR_CUTOFF),
            "n_obstacles": len(self._active_obs),
            "layout": str(self.layout),
            "layout_kind": str(self._layout_kind),
            "expert_action": expert_a,
            "geodesic": float(geo) if np.isfinite(geo) else -1.0,
            "expert_unreachable": unreachable,
        }
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        action = int(action)
        prev_action = int(self.prev_action)
        expert_a = self.expert_action() if self._expert is not None else -1
        name = ACTION_NAMES[action]
        dx, dy, dyaw = PRIMITIVES[name]
        x0, y0 = float(self.data.qpos[0]), float(self.data.qpos[1])
        yaw0 = self._pelvis_yaw()
        c, s = np.cos(yaw0), np.sin(yaw0)
        x1 = x0 + c * dx - s * dy
        y1 = y0 + s * dx + c * dy
        yaw1 = yaw0 + dyaw

        collided = False
        last_good = (x0, y0, yaw0)
        n = max(1, self.primitive_substeps)
        for k in range(1, n + 1):
            a = k / n
            self._set_root(x0 + a * (x1 - x0), y0 + a * (y1 - y0), yaw0 + a * (yaw1 - yaw0))
            if self._colliding():
                collided = True
                self._set_root(*last_good)
                break
            last_good = (float(self.data.qpos[0]), float(self.data.qpos[1]), self._pelvis_yaw())

        if collided:
            self._collision_streak += 1
        else:
            self._collision_streak = 0

        self.prev_action = action
        self.step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward(collided, expert_a, action, prev_action)
        terminated = bool(info["arrived"])
        truncated = (self.step_count >= self.max_episode_steps) or (
            self._collision_streak >= self.collision_streak_limit
        )
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.track_renderer is None:
                self.track_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.track_renderer.update_scene(self.data, camera="arena")
            return self.track_renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.track_renderer is not None:
            self.track_renderer.close()
            self.track_renderer = None
        self.perception.close()

    def head_rgb(self) -> np.ndarray:
        return self.perception.rgb(self.data)
