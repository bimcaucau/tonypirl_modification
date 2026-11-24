"""Unit tests for observation contract, transforms, and offline simulation.

Matches SIM2REAL_NAV.md section 10:
  1. build_obs vec length 18, occupancy shape, one-hot of 5 has a 1 at index 5.
  2. dist=3, heading=0 -> vec[0]==1, vec[1]==0, vec[2]==1.
  3. scale_range(999, 1.2)==1.0, scale_range(0.6, 1.2)==0.5.
  4. Fake loop: FakeHardware + zeros occupancy + fixed-body goal (2,0) + scripted actions
     updates dead-reckon pose using sim primitives: one go_forward -> x~0.08.
  5. If the zip is on disk, load_nav_policy + synthetic-clear returns an int in 0..6.
  6. Camera optical axis transform test: point 1m in front of camera (+Z optical) lands near body +X.
"""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.constants import (
    ACTION_NAMES,
    GOAL_DIST_SCALE,
    OCCUPANCY_SIZE,
    SIM_PRIMITIVES,
    SONAR_CUTOFF,
    TOF_CUTOFF,
)
from sim2real_implementation.goal import GoalSource, transform_cam_to_body
from sim2real_implementation.hardware import FakeHardware
from sim2real_implementation.observations import (
    build_obs,
    gravity_from_imu,
    one_hot,
    polar_goal,
    scale_range,
)
from sim2real_implementation.policy import act, inspect_checkpoint, load_nav_policy


class TestObsContract(unittest.TestCase):
    """Test suite validating observation contract, goal math, and hardware mocking."""

    def test_1_build_obs_shapes_and_one_hot(self):
        """Test 1: build_obs vec length 18, occupancy shape, one-hot of 5 has a 1 at index 5."""
        occ_zeros = np.zeros((OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32)
        obs = build_obs(
            dist=1.5,
            heading=0.0,
            tof_m=1.0,
            sonar_m=2.5,
            gravity=[0.0, 0.0, -1.0],
            prev_action=5,
            occupancy=occ_zeros,
        )
        self.assertIn("vec", obs)
        self.assertIn("occupancy", obs)
        self.assertEqual(obs["vec"].shape, (18,))
        self.assertEqual(obs["occupancy"].shape, (1, 64, 64))

        # Check one-hot encoding for action 5
        oh = one_hot(5)
        self.assertEqual(oh.shape, (7,))
        self.assertEqual(oh[5], 1.0)
        self.assertEqual(float(np.sum(oh)), 1.0)

        # In 18-D vec: prev_action one-hot occupies indices [11:18]
        # Action 5 should be at index 11 + 5 = 16
        self.assertEqual(obs["vec"][16], 1.0)
        self.assertEqual(float(np.sum(obs["vec"][11:18])), 1.0)

    def test_2_polar_goal(self):
        """Test 2: dist=3, heading=0 -> vec[0]==1, vec[1]==0, vec[2]==1."""
        pg = polar_goal(dist=3.0, heading=0.0)
        self.assertAlmostEqual(pg[0], 1.0, places=5)
        self.assertAlmostEqual(pg[1], 0.0, places=5)
        self.assertAlmostEqual(pg[2], 1.0, places=5)

        # Test within full build_obs
        obs = build_obs(
            dist=3.0,
            heading=0.0,
            tof_m=1.2,
            sonar_m=5.0,
            gravity=[0.0, 0.0, -1.0],
            prev_action=0,
            occupancy=np.zeros((1, 64, 64), dtype=np.float32),
        )
        self.assertAlmostEqual(obs["vec"][0], 1.0, places=5)
        self.assertAlmostEqual(obs["vec"][1], 0.0, places=5)
        self.assertAlmostEqual(obs["vec"][2], 1.0, places=5)

    def test_3_scale_range(self):
        """Test 3: scale_range(999, 1.2)==1.0, scale_range(0.6, 1.2)==0.5."""
        self.assertAlmostEqual(scale_range(999.0, 1.2), 1.0, places=5)
        self.assertAlmostEqual(scale_range(0.6, 1.2), 0.5, places=5)
        self.assertAlmostEqual(scale_range(-5.0, 1.2), 1.0, places=5)  # Invalid/negative clipped/miss
        self.assertAlmostEqual(scale_range(float("nan"), 1.2), 1.0, places=5)
        self.assertAlmostEqual(scale_range(float("inf"), 1.2), 1.0, places=5)

        # Also verify sonar cutoff
        self.assertAlmostEqual(scale_range(99999.0, 5.0), 1.0, places=5)
        self.assertAlmostEqual(scale_range(2.5, 5.0), 0.5, places=5)

    def test_4_fake_loop_deadreckon(self):
        """Test 4: Fake loop updates dead-reckon pose using sim primitives: one go_forward -> x~0.08."""
        hw = FakeHardware()
        hw.start()

        goal = GoalSource(mode="deadreckon", initial_goal=(2.0, 0.0))
        self.assertEqual(goal.x, 0.0)
        self.assertEqual(goal.y, 0.0)
        self.assertEqual(goal.yaw, 0.0)

        # Script action: go_forward
        scripted_actions = ["go_forward", "go_forward", "turn_left"]
        for act_name in scripted_actions:
            hw.play(act_name)
            goal.on_action(act_name)

        # After first go_forward: x ~ 0.08
        # After two go_forward: x ~ 0.16
        # After turn_left: yaw ~ 0.35 rad
        self.assertAlmostEqual(goal.x, 0.16, places=4)
        self.assertAlmostEqual(goal.y, 0.00, places=4)
        self.assertAlmostEqual(goal.yaw, 0.35, places=4)

        # Verify current body goal
        dist, heading, gx, gy = goal.current()
        self.assertAlmostEqual(dist, 2.0 - 0.16, places=3)
        # Robot turned left 0.35 rad, so goal is now to its right (heading < 0)
        self.assertAlmostEqual(heading, -0.35, places=3)

    def test_5_policy_load_and_synthetic_predict(self):
        """Test 5: If the zip is on disk, load_nav_policy + synthetic-clear returns an int in 0..6."""
        model_path = os.path.join(REPO_ROOT, "models_ckpt/new_expert_implementation/tonypi_final_model.zip")
        if not os.path.exists(model_path):
            self.skipTest(f"Model checkpoint not found at {model_path}")

        meta = inspect_checkpoint(model_path)
        self.assertTrue(meta["valid"], f"Checkpoint invalid: {meta['reason']}")

        model = load_nav_policy(model_path, device="cpu")
        self.assertIsNotNone(model)

        obs_clear = build_obs(
            dist=2.0,
            heading=0.0,
            tof_m=1.2,
            sonar_m=5.0,
            gravity=[0.0, 0.0, -1.0],
            prev_action=0,
            occupancy=np.zeros((1, 64, 64), dtype=np.float32),
        )
        action = act(model, obs_clear, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, 6)
        # On a clear straight path, policy chooses go_forward (1)
        self.assertEqual(action, 1)

    def test_6_optical_axis_cam_to_body_transform(self):
        """Test 6: Point 1m in front of camera (+Z optical axis) must land near body +X."""
        # 1m directly along camera optical axis (in OpenCV: X=0, Y=0, Z=1.0)
        p_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        p_body = transform_cam_to_body(p_cam)

        # Expected: body X = 1.0 + 0.08 = 1.08, body Y = 0.0, body Z = 0.04
        self.assertGreater(p_body[0], 0.9, "Optical axis point did not land at body +X!")
        self.assertAlmostEqual(p_body[0], 1.08, places=4)
        self.assertAlmostEqual(p_body[1], 0.00, places=4)
        self.assertAlmostEqual(p_body[2], 0.04, places=4)

    def test_7_safety_checks(self):
        """Test 7: Safety layer handles estop, arrival, max-steps, fall, and ToF guard."""
        from sim2real_implementation.safety import Safety

        hw = FakeHardware()
        safety = Safety(
            max_steps=10,
            max_seconds=30.0,
            arrive_radius=0.20,
            fall_deg=45.0,
            tof_guard=0.15,
        )

        # 1. Normal clear step
        act_id, act_name, reason = safety.filter_action(
            action_idx=1, hw=hw, dist=1.5, step=0, elapsed_s=0.1, tof_m=1.0
        )
        self.assertEqual(act_id, 1)
        self.assertEqual(act_name, "go_forward")
        self.assertIsNone(reason)

        # 2. ToF guard triggers on go_forward if ToF < 0.15m
        act_id, act_name, reason = safety.filter_action(
            action_idx=1, hw=hw, dist=1.5, step=1, elapsed_s=0.5, tof_m=0.10
        )
        self.assertEqual(act_name, "back_fast")
        self.assertEqual(reason, "tof_guard")

        # 3. Arrived at goal
        act_id, act_name, reason = safety.filter_action(
            action_idx=1, hw=hw, dist=0.15, step=2, elapsed_s=1.0, tof_m=1.0
        )
        self.assertEqual(act_name, "stand")
        self.assertEqual(reason, "arrived")

        # 4. Max steps reached
        act_id, act_name, reason = safety.filter_action(
            action_idx=1, hw=hw, dist=1.5, step=10, elapsed_s=1.5, tof_m=1.0
        )
        self.assertEqual(act_name, "stand")
        self.assertEqual(reason, "max_steps")

        # 5. Fall detection (3 cycles of extreme tilt)
        hw_fall = FakeHardware(accel=np.array([0.0, 0.0, 1.0]))  # |atan2(0, 1)| = 0 deg < 45 deg -> fall forward
        safety_fall = Safety(fall_deg=45.0, recover_fall=True)
        # Cycle 1 & 2: accumulating
        safety_fall.filter_action(1, hw_fall, 1.5, 0, 0.1, 1.0)
        safety_fall.filter_action(1, hw_fall, 1.5, 1, 0.2, 1.0)
        # Cycle 3: fall detected
        _, act_to_play, reason = safety_fall.filter_action(1, hw_fall, 1.5, 2, 0.3, 1.0)
        self.assertEqual(reason, "fall_recover")
        self.assertEqual(act_to_play, "stand_up_front")

        # 6. E-stop
        safety_estop = Safety()
        safety_estop.estop = True
        act_id, act_name, reason = safety_estop.filter_action(1, hw, 1.5, 0, 0.1, 1.0)
        self.assertEqual(act_name, "stand")
        self.assertEqual(reason, "estop")

    def test_8_occupancy_encoders(self):
        """Test 8: Occupancy encoders generate correct (1, 64, 64) binary float32 grids."""
        from sim2real_implementation.occupancy_real import OccupancyEncoder, color_occupancy

        # Zeros mode
        enc_zeros = OccupancyEncoder(mode="zeros")
        occ_z = enc_zeros.encode(None)
        self.assertEqual(occ_z.shape, (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE))
        self.assertEqual(float(np.sum(occ_z)), 0.0)

        # Color mode with mock red box
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw red obstacle in center (BGR: 0, 0, 255)
        img[150:350, 200:440] = [0, 0, 255]
        occ_c = color_occupancy(img, hsv_lo=[0, 80, 80], hsv_hi=[12, 255, 255])
        self.assertEqual(occ_c.shape, (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE))
        self.assertGreater(float(np.sum(occ_c)), 50.0)
        self.assertTrue(set(np.unique(occ_c)).issubset({0.0, 1.0}))

        # Hue wrap: H_lo > H_hi (red wraps around 179)
        from sim2real_implementation.occupancy_real import hsv_in_range

        hsv_img = np.zeros((8, 8, 3), dtype=np.uint8)
        hsv_img[:] = (175, 200, 200)
        wrap = hsv_in_range(hsv_img, [170, 80, 80], [12, 255, 255])
        self.assertEqual(int(wrap.max()), 255)
        no_wrap = hsv_in_range(hsv_img, [0, 80, 80], [12, 255, 255])
        self.assertEqual(int(no_wrap.max()), 0)

    def test_9_encode_views_three_panes(self):
        """encode_views keeps (1,64,64) occupancy and returns camera/overlay/binary panes."""
        from sim2real_implementation.occupancy_real import OccupancyEncoder

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[150:350, 200:440] = [0, 0, 255]
        enc = OccupancyEncoder(mode="color", hsv_lo=[0, 80, 80], hsv_hi=[12, 255, 255])
        views = enc.encode_views(img)
        self.assertEqual(views["occupancy"].shape, (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE))
        self.assertEqual(views["camera"].shape, (480, 640, 3))
        self.assertEqual(views["overlay"].shape, (480, 640, 3))
        self.assertEqual(views["binary"].shape, (480, 640, 3))
        self.assertEqual(views["full_mask"].shape, (480, 640))
        self.assertGreater(int(views["full_mask"].sum()), 0)
        # Policy encode() must still be the occupancy tensor only
        occ = enc.encode(img)
        self.assertEqual(occ.shape, (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE))
        np.testing.assert_array_equal(occ, views["occupancy"])
        # Binary pane: obstacle red-ish, road green-ish
        cy, cx = 250, 320
        self.assertGreater(int(views["binary"][cy, cx, 2]), int(views["binary"][cy, cx, 1]))
        self.assertGreater(int(views["binary"][10, 10, 1]), int(views["binary"][10, 10, 2]))

    def test_10_tag_lost_stand_and_clamping(self):
        """test 10: tag-lost stand forces stand, hold preserves goal, and one_hot clamps index."""
        # 1. Goal tag-lost stand logic
        goal_stand = GoalSource(mode="tag", initial_goal=(1.5, 0.0), tag_lost="stand")
        # Initially not visible
        self.assertFalse(goal_stand.tag_visible)
        self.assertTrue(goal_stand.should_force_stand())

        # Simulate tag detection
        goal_stand.tag_visible = True
        self.assertFalse(goal_stand.should_force_stand())

        # Tag lost again
        goal_stand.tag_visible = False
        self.assertTrue(goal_stand.should_force_stand())

        # Tag-lost hold policy
        goal_hold = GoalSource(mode="tag", initial_goal=(1.5, 0.0), tag_lost="hold")
        self.assertFalse(goal_hold.tag_visible)
        self.assertFalse(goal_hold.should_force_stand())
        # Current returns last known goal
        dist, heading, gx, gy = goal_hold.current()
        self.assertAlmostEqual(dist, 1.5, places=3)

        # 2. one_hot index clamping hygiene
        oh_hi = one_hot(10)
        self.assertEqual(oh_hi.shape, (7,))
        self.assertEqual(oh_hi[6], 1.0)

        oh_neg = one_hot(-3)
        self.assertEqual(oh_neg.shape, (7,))
        self.assertEqual(oh_neg[0], 1.0)

    def test_11_vl53l0x_import_from_functions_not_cwd(self):
        """Stock VL53L0X.py CDLL('./vl53l0x_python.so') must load from Functions/, not process CWD."""
        import tempfile
        from sim2real_implementation.hardware import find_tonypi_home, import_vl53l0x

        with tempfile.TemporaryDirectory() as tmp:
            functions = os.path.join(tmp, "TonyPi", "Functions")
            os.makedirs(functions)
            so_path = os.path.join(functions, "vl53l0x_python.so")
            open(so_path, "wb").close()
            with open(os.path.join(functions, "VL53L0X.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import os\n"
                    "CWD_AT_IMPORT = os.getcwd()\n"
                    "class Vl53l0xAccuracyMode:\n"
                    "    BETTER = 1\n"
                    "class VL53L0X:\n"
                    "    def __init__(self, **kw): pass\n"
                    "    def open(self): pass\n"
                    "    def start_ranging(self, mode=1): pass\n"
                    "    def get_distance(self): return 500\n"
                )
            old_env = os.environ.get("TONYPI_HOME")
            os.environ["TONYPI_HOME"] = os.path.join(tmp, "TonyPi")
            try:
                home = find_tonypi_home()
                self.assertEqual(os.path.abspath(home), os.path.abspath(os.path.join(tmp, "TonyPi")))
                outside = os.path.join(tmp, "elsewhere")
                os.makedirs(outside)
                prev = os.getcwd()
                os.chdir(outside)
                try:
                    mod = import_vl53l0x(functions)
                    self.assertEqual(os.path.abspath(mod.CWD_AT_IMPORT), os.path.abspath(functions))
                    self.assertEqual(os.path.abspath(os.getcwd()), os.path.abspath(outside))
                    self.assertEqual(mod.VL53L0X().get_distance(), 500)
                finally:
                    os.chdir(prev)
            finally:
                sys.modules.pop("VL53L0X", None)
                if old_env is None:
                    os.environ.pop("TONYPI_HOME", None)
                else:
                    os.environ["TONYPI_HOME"] = old_env

    def test_12_refuse_detect_checkpoint(self):
        """Detect-only weights (yolov5n, onboard best.pt) must fail before YOLO load."""
        from tonypi_rl.perception import refuse_detect_checkpoint

        with self.assertRaises(RuntimeError):
            refuse_detect_checkpoint("yolov5n.pt")
        with self.assertRaises(RuntimeError):
            refuse_detect_checkpoint("/home/pi/TonyPi/Functions/model/best.pt")
        with self.assertRaises(RuntimeError):
            refuse_detect_checkpoint("yolo26n.pt")  # detect, missing -seg
        refuse_detect_checkpoint("yolo26n-seg.pt")
        refuse_detect_checkpoint("yolo11n-seg.pt")
        refuse_detect_checkpoint("yolo26n-sem-ade20k.pt")
        refuse_detect_checkpoint("yolo26n-sem.pt")

    def test_13_semantic_table_legs_vs_floor(self):
        """ADE20K semantic map: floor free, table (incl. legs region) occupied."""
        from tonypi_rl.perception import semantic_obstacle_mask

        names = {0: "wall", 3: "floor", 12: "person", 15: "table", 19: "chair"}
        h, w = 40, 40
        cmap = np.full((h, w), 3, dtype=np.int32)  # floor
        cmap[:, :3] = 0  # wall
        cmap[8:36, 12:28] = 15  # table top + legs as one stuff class
        cmap[22:38, 2:9] = 19  # chair
        full, labels = semantic_obstacle_mask(cmap, names)
        self.assertEqual(int(full[0, 20]), 0)  # floor
        self.assertEqual(int(full[2, 1]), 0)  # wall
        self.assertEqual(int(full[20, 20]), 255)  # table body
        self.assertEqual(int(full[34, 14]), 255)  # table "leg" pixels
        self.assertEqual(int(full[30, 5]), 255)  # chair
        found = {n for n, _, _ in labels}
        self.assertIn("table", found)
        self.assertIn("chair", found)
        self.assertNotIn("floor", found)
        self.assertNotIn("wall", found)

    def test_14_relative_goal_polar(self):
        """--dist/--angle-deg matches body +x forward, +y left."""
        from sim2real_implementation.goal import body_goal_from_polar
        from sim2real_implementation.go_to import goal_xy, parse_args

        x, y = body_goal_from_polar(1.0, 0.0)
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)
        x, y = body_goal_from_polar(1.0, 90.0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)
        x, y = body_goal_from_polar(1.0, -90.0)
        self.assertAlmostEqual(y, -1.0, places=5)
        ns = parse_args(["--dist", "0.5", "--angle-deg", "0", "--backend", "fake"])
        self.assertEqual(goal_xy(ns)[0], 0.5)
        with self.assertRaises(SystemExit):
            parse_args(["--xy", "1", "0", "--angle-deg", "10"])


if __name__ == "__main__":
    unittest.main()
