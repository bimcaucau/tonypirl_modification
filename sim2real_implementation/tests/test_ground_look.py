"""Ground-look geometry, veto, and dead-reckon correction. No robot, no policy."""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.goal import GoalSource
from sim2real_implementation.ground_look import (
    DEFAULT_LOOK_PULSE,
    FOOTPRINTS,
    GroundLook,
    LookResult,
    aim_pulses,
    avoid_action,
    floor_shift_m,
    fused_tof,
    ground_points,
    intrinsics_for,
    pitch_down_rad,
    shift_to_motion,
)
from sim2real_implementation.hardware import FakeHardware


def _K(w=640, h=480):
    return intrinsics_for(w, h)


class TestGroundGeometry(unittest.TestCase):
    def test_pitch_sign_flips_which_side_of_center_is_down(self):
        down = pitch_down_rad(900)
        flipped = pitch_down_rad(2100, pitch_sign=1)
        self.assertAlmostEqual(down, flipped, places=5)
        self.assertLess(pitch_down_rad(2100), 0.0)

    def test_level_center_ray_misses_the_floor(self):
        K = _K()
        _, _, valid = ground_points(np.array([K[2]]), np.array([K[3]]), 0.0, 0.0, K)
        self.assertFalse(bool(valid[0]))

    def test_look_down_center_hits_in_front(self):
        K = _K()
        pitch = pitch_down_rad(DEFAULT_LOOK_PULSE)
        # Steeper than ~42° puts the shins in frame and every step is vetoed.
        self.assertLess(pitch, np.deg2rad(42))
        self.assertGreater(pitch, np.deg2rad(25))
        xs, _, valid = ground_points(np.array([K[2]]), np.array([K[3]]), pitch, 0.0, K)
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(xs[0]), 0.45)
        self.assertLess(float(xs[0]), 0.80)
        # Bottom of the image is still floor in front of the feet, not the feet.
        bottom, _, bottom_ok = ground_points(np.array([K[2]]), np.array([470.0]), pitch, 0.0, K)
        self.assertTrue(bool(bottom_ok[0]))
        self.assertGreater(float(bottom[0]), 0.22)

    def test_lower_pixel_is_closer(self):
        K = _K()
        pitch = pitch_down_rad(DEFAULT_LOOK_PULSE)
        us = np.array([K[2], K[2]])
        vs = np.array([K[3], K[3] + 40])
        xs, _, valid = ground_points(us, vs, pitch, 0.0, K)
        self.assertTrue(bool(valid.all()))
        self.assertLess(float(xs[1]), float(xs[0]))

    def test_texture_moving_down_means_forward(self):
        K = _K()
        pitch = pitch_down_rad(DEFAULT_LOOK_PULSE)
        dx, dy = shift_to_motion((0.0, 12.0), pitch, K, 640, 480)
        self.assertGreater(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0, places=3)
        dx_right, dy_right = shift_to_motion((10.0, 0.0), pitch, K, 640, 480)
        # Image right is body -Y, so texture moving right means the robot went left.
        self.assertGreater(dy_right, 0.0)
        self.assertAlmostEqual(dx_right, 0.0, places=3)


class TestFusedRange(unittest.TestCase):
    def test_closer_of_beam_and_floor(self):
        self.assertAlmostEqual(fused_tof(1.2, None), 1.2)
        self.assertAlmostEqual(fused_tof(1.2, 0.30), 0.30)
        self.assertAlmostEqual(fused_tof(0.10, 0.50), 0.10)
        # A single noisy hit must not become "already colliding".
        self.assertAlmostEqual(fused_tof(1.2, 0.01), 0.08)


class TestVeto(unittest.TestCase):
    def test_avoid_clear_footprint_keeps_the_step(self):
        blocked = {name: False for name in FOOTPRINTS}
        name, reason = avoid_action("go_forward", blocked, "left")
        self.assertEqual(name, "go_forward")
        self.assertIsNone(reason)

    def test_avoid_wire_ahead_does_not_step_on_it(self):
        blocked = {name: False for name in FOOTPRINTS}
        blocked["go_forward"] = True
        blocked["turn_left"] = True
        name, reason = avoid_action("go_forward", blocked, "right")
        self.assertEqual(reason, "ground_look")
        self.assertEqual(name, "turn_right")

    def test_uniform_floor_is_not_blocked(self):
        bgr = np.full((480, 640, 3), 160, dtype=np.uint8)
        hw = FakeHardware(bgr=bgr)
        res = GroundLook(sweep=False).survey(hw, encode_views=None, action_name="go_forward")
        self.assertFalse(res.blocked.get("go_forward", False))
        self.assertIsNone(res.forward_range_m)
        self.assertEqual(hw.head_pitch_pulse, 1500)
        self.assertEqual(hw.head_yaw_pulse, 1500)

    def test_wire_in_the_step_blocks_forward(self):
        bgr = np.full((480, 640, 3), 180, dtype=np.uint8)
        pitch = pitch_down_rad(DEFAULT_LOOK_PULSE)
        K = _K()
        vs = np.arange(0, 480, 2)
        us = np.full(vs.shape, K[2])
        xs, _, valid = ground_points(us, vs, pitch, 0.0, K)
        x0, x1, _, _ = FOOTPRINTS["go_forward"]
        rows = vs[valid & (xs > x0) & (xs < x1)]
        self.assertGreater(rows.size, 0, "look-down view does not see the forward footprint")
        v = int(np.median(rows))
        bgr[v - 1 : v + 2, 160:480] = (20, 20, 20)
        hw = FakeHardware(bgr=bgr)
        res = GroundLook(sweep=False).survey(hw, encode_views=None, action_name="go_forward")
        self.assertTrue(res.blocked.get("go_forward"), res.fractions)
        self.assertIsNotNone(res.forward_range_m)
        self.assertGreater(res.forward_range_m, 0.10)
        self.assertLess(res.forward_range_m, 0.50)
        safe, reason = res.avoid("go_forward")
        self.assertEqual(reason, "ground_look")
        self.assertNotEqual(safe, "go_forward")
        self.assertEqual(hw.head_pitch_pulse, 1500)

    def test_mislabeled_floor_does_not_veto(self):
        """Semantic mask covering the whole frame is a bad floor label, not a wire."""
        bgr = np.full((240, 320, 3), 150, dtype=np.uint8)
        full = np.full((240, 320), 255, dtype=np.uint8)

        def views(_bgr):
            return {"full_mask": full}

        res = GroundLook(sweep=False).survey(FakeHardware(bgr=bgr), views, "go_forward")
        self.assertTrue(res.semantic_saturated)
        self.assertFalse(res.blocked.get("go_forward", False))

    def test_small_semantic_block_is_a_range(self):
        bgr = np.full((480, 640, 3), 170, dtype=np.uint8)
        pitch = pitch_down_rad(DEFAULT_LOOK_PULSE)
        K = _K()
        vs, us = np.mgrid[0:480:2, 0:640:2]
        xs, ys, valid = ground_points(us.ravel(), vs.ravel(), pitch, 0.0, K)
        x0, x1, y0, y1 = FOOTPRINTS["go_forward"]
        inside = valid & (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1)
        self.assertGreater(int(inside.sum()), 30)
        full = np.zeros((480, 640), dtype=np.uint8)
        # Every 4th in-box sample: a real object, not a saturated floor.
        pick = np.flatnonzero(inside)[::4]
        full[vs.ravel()[pick], us.ravel()[pick]] = 255

        def views(_bgr):
            return {"full_mask": full}

        res = GroundLook(sweep=False).survey(FakeHardware(bgr=bgr), views, "go_forward")
        self.assertTrue(res.blocked.get("go_forward"), res.fractions)
        self.assertIsNotNone(res.forward_range_m)

    def test_sweep_looks_left_and_right_then_recenters(self):
        bgr = np.full((120, 160, 3), 140, dtype=np.uint8)
        hw = FakeHardware(bgr=bgr)
        res = GroundLook(sweep=True).survey(hw, None, "go_forward")
        self.assertEqual(res.poses, ["center", "head_left", "left", "right", "head_right"])
        self.assertEqual((hw.head_pitch_pulse, hw.head_yaw_pulse), (1500, 1500))
        yaws = [move[1] for move in hw.head_moves]
        self.assertGreater(max(yaws), 1500)
        self.assertLess(min(yaws), 1500)
        # Side moves are most of a right angle, not a 250 µs glance, so the
        # servo is given more than the old 400 ms.
        side_moves = [m for m in hw.head_moves if abs(m[1] - 1500) > 400]
        self.assertGreaterEqual(len(side_moves), 2)
        # The head turn from center is most of a right angle and must be given
        # longer than the old 400 ms. A later pose already near that yaw is short.
        self.assertGreaterEqual(max(m[2] for m in side_moves), 600)

    def test_side_look_centers_on_the_strafe_footprint(self):
        """A wire beside the foot is a strafe problem. The optical axis of the
        left pose has to land in that step, not 30 cm ahead of it."""
        look = GroundLook(sweep=True)
        poses = {label: (pitch, yaw) for label, pitch, yaw in look._poses_for("left_move_fast")}
        self.assertIn("left", poses)
        self.assertIn("right", poses)
        # Retreat used to skip the sweep. A blocked back-up turns, so it needs the sides too.
        self.assertEqual(
            [label for label, _, _ in look._poses_for("back_fast")],
            ["center", "head_left", "left", "right", "head_right"],
        )
        K = intrinsics_for(640, 480)
        cx, cy = K[2], K[3]
        for label, box_name, y_sign in (
            ("left", "left_move_fast", 1.0),
            ("right", "right_move_fast", -1.0),
        ):
            pitch_pulse, yaw_pulse = poses[label]
            pitch = pitch_down_rad(pitch_pulse, pitch_sign=-1)
            yaw = math.radians((yaw_pulse - 1500) * 0.09)
            xs, ys, valid = ground_points([cx], [cy], pitch, yaw, K)
            self.assertTrue(bool(valid[0]), label)
            x0, x1, y0, y1 = FOOTPRINTS[box_name]
            self.assertGreater(float(xs[0]), x0)
            self.assertLess(float(xs[0]), x1)
            self.assertGreater(float(ys[0]) * y_sign, 0.0)
            inner, outer = sorted((abs(y0), abs(y1)))
            self.assertGreater(abs(float(ys[0])), inner)
            self.assertLess(abs(float(ys[0])), outer)
        # The aim helper and the pose agree, including a flipped yaw servo.
        flipped = aim_pulses(0.55, 0.20, pitch_sign=-1, yaw_sign=-1)
        self.assertLess(flipped[1], 1500)
        self.assertEqual(flipped[0], poses["left"][0])

    def test_head_turn_looks_beside_the_body_and_blocks_that_side(self):
        """Turning the head, not only looking at the toes. The optical axis lands
        well outside the strafe step, and clutter there blocks the turn and the
        strafe but not a straight step."""
        look = GroundLook(sweep=True)
        poses = {label: (pitch, yaw) for label, pitch, yaw in look._poses_for("go_forward")}
        K = intrinsics_for(640, 480)
        cx, cy = K[2], K[3]
        for label, y_sign in (("head_left", 1.0), ("head_right", -1.0)):
            pitch_pulse, yaw_pulse = poses[label]
            self.assertGreater(abs(yaw_pulse - 1500), abs(poses["left"][1] - 1500))
            pitch = pitch_down_rad(pitch_pulse, pitch_sign=-1)
            yaw = math.radians((yaw_pulse - 1500) * 0.09)
            xs, ys, valid = ground_points([cx], [cy], pitch, yaw, K)
            self.assertTrue(bool(valid[0]), label)
            self.assertGreater(abs(float(ys[0])), 0.40)
            self.assertGreater(float(xs[0]), 0.35)
            self.assertLess(pitch, math.radians(42))
            self.assertGreater(float(ys[0]) * y_sign, 0.0)

        res = LookResult()
        ahead = np.full(80, 0.25)
        aside = np.full(40, 0.30)
        look._summarize(
            res,
            [np.concatenate([ahead, np.full(40, 0.40)])],
            [np.concatenate([np.zeros(80), np.full(40, 0.40)])],
            [aside],
            [np.full(40, 0.40)],
            [],
            [],
        )
        self.assertTrue(res.blocked.get("turn_left"))
        self.assertTrue(res.blocked.get("left_move_fast"))
        self.assertFalse(res.blocked.get("go_forward"))
        self.assertFalse(res.blocked.get("turn_right"))
        self.assertEqual(res.clearer_side, "right")


class TestLiveViewEpoch(unittest.TestCase):
    def test_invalidate_discards_a_frame_grabbed_before_it(self):
        from sim2real_implementation.view_segmentation import LiveTriple

        state = {"n": 0, "hold": False}

        def read():
            while state["hold"]:
                time_sleep_short()
            state["n"] += 1
            img = np.full((48, 64, 3), state["n"], dtype=np.uint8)
            return img

        class Enc:
            def encode_views(self, bgr):
                z = np.zeros(bgr.shape[:2], dtype=np.uint8)
                return {
                    "occupancy": np.zeros((1, 64, 64), dtype=np.float32),
                    "full_mask": z,
                    "overlay": bgr,
                    "binary": bgr,
                    "camera": bgr,
                    "labels": [],
                }

        live = LiveTriple(Enc(), 0)
        live.start(read, lambda: "")
        try:
            first = live.wait_views(2.0)
            self.assertIsNotNone(first)
            seen = int(first["camera"][0, 0, 0])
            state["hold"] = True
            live.invalidate()
            state["hold"] = False
            nxt = live.wait_views(2.0)
            self.assertIsNotNone(nxt)
            self.assertGreater(int(nxt["camera"][0, 0, 0]), seen)
        finally:
            live.stop()


def time_sleep_short():
    import time
    time.sleep(0.01)


class TestFloorShift(unittest.TestCase):
    def test_downward_image_shift_is_a_forward_step(self):
        rng = np.random.default_rng(1)
        prev = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        curr = np.roll(prev, 10, axis=0)  # texture moved down
        K = intrinsics_for(320, 240)
        shift = floor_shift_m(prev, curr, pitch_down_rad(DEFAULT_LOOK_PULSE), K)
        self.assertIsNotNone(shift)
        dx, dy, response = shift
        self.assertGreater(response, 0.5)
        self.assertGreater(dx, 0.0)
        self.assertLess(abs(dy), abs(dx))

    def test_blank_floor_is_not_a_measurement(self):
        blank = np.full((240, 320, 3), 128, dtype=np.uint8)
        K = intrinsics_for(320, 240)
        self.assertIsNone(floor_shift_m(blank, blank, pitch_down_rad(DEFAULT_LOOK_PULSE), K))


class TestDeadReckonCorrection(unittest.TestCase):
    def test_measured_step_replaces_the_primitive(self):
        goal = GoalSource(mode="deadreckon", initial_goal=(2.0, 0.0))
        goal.on_action("go_forward")
        self.assertAlmostEqual(goal.x, 0.08, places=5)
        self.assertTrue(goal.correct_last_translation(0.03, 0.0))
        self.assertAlmostEqual(goal.x, 0.03, places=5)
        # Implausible measurements are ignored, and a turn has no translation to replace.
        self.assertFalse(goal.correct_last_translation(0.9, 0.0))
        self.assertAlmostEqual(goal.x, 0.03, places=5)
        goal.on_action("turn_left")
        self.assertFalse(goal.correct_last_translation(0.04, 0.0))


if __name__ == "__main__":
    unittest.main()
