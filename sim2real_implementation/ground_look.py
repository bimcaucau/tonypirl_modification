"""Look at the ground before a step, and measure distance from that look.

The policy camera is held level (pulse 1500) so its occupancy stays in the
training distribution. At that pitch the bottom of a ~60° vertical image meets
the floor about half a metre out, and the chest-height VL53L0X beam passes
over wires and small blocks. Those are exactly what the robot trips on.

This module pitches the head down, reads one or three views, and:

  1. Projects clutter pixels onto the floor with a pinhole + known pitch.
     If the next step's footprint contains clutter, the proposed clip is
     replaced (`avoid_action`). The head is back at center before the next
     policy observation.
  2. Reports the nearest clutter range in the forward corridor so the caller
     can put min(ToF, that range) into vec[3]. ToF alone reads "clear" when
     the beam flies over a low object.
  3. Phase-correlates successive look-down frames. A pure translation moves
     floor texture by a known metres-per-pixel, which corrects the dead-reckon
     step length (the 8 cm primitive is a guess).

Geometry (pelvis body frame, +X forward, +Y left, +Z up), from tonypi.xml:

  pelvis origin is 0.267 m above the floor
  camera at pitch 0 is (0.08, 0, 0.110) in the pelvis frame
  OpenCV camera: +Z optical, +X right, +Y down
  body_from_cam at pitch 0 matches goal.R_BODY_FROM_CAM

Servo 1: hand_nod.py comments pulse (center - amp) as down. Center 1500 is
level, matching HARDWARE.md. Degrees = (1500 - pulse) * 0.09, from the
500..2500 µs = 180° Hiwonder mapping. Confirm on the robot; if the head looks
up, the rays miss the floor and the veto stays silent rather than firing.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Pelvis-frame camera at head pitch 0, and the floor under it.
PELVIS_HEIGHT_M = 0.267
CAM_POS_BODY = np.array([0.08, 0.0, 0.110], dtype=np.float64)  # neck+head+cam offset
GROUND_Z = -PELVIS_HEIGHT_M

CENTER_PULSE = 1500
DEG_PER_PULSE = 180.0 / 2000.0  # 500..2500 µs spans 180°
PULSE_MIN = 600
PULSE_MAX = 2400

# 36° down (pulse 1100). The old 54° (pulse 900) put the shins in the bottom of
# the frame; those pixels were scored as a wire and every step became back_fast.
# At 36° the bottom ray meets the floor ~0.26 m ahead, past the feet, and the
# image center is ~0.60 m ahead.
DEFAULT_LOOK_PULSE = 1100
# Old side glance was ±250 µs (~22°). That leaves the strafe footprint in the
# bottom corner of the forward view. Side poses are aimed instead; this value
# is only kept so a yaml key from before that change still loads.
DEFAULT_YAW_AMP = 250
# Same pitch as the forward look (~36°), yawed onto the floor ahead-and-aside.
# The old points (0.16, 0.13) were the shin, and the robot backed up forever.
SIDE_AIM_M = {
    "left": (0.55, 0.20),
    "right": (0.55, -0.20),
}
# Head turned aside, but no steeper than the forward look. A closer aim frames
# the shoulder and that reads as an obstacle.
HEAD_PAN_AIM_M = {
    "head_left": (0.55, 0.55),
    "head_right": (0.55, -0.55),
}
# Outside the strafe footprint. Near edge stays past the feet. Clutter here
# blocks the turn and the strafe toward that side, not a straight step.
SIDE_CORRIDORS = {
    "left": (0.28, 0.75, 0.22, 0.70),
    "right": (0.28, 0.75, -0.70, -0.22),
}
SIDE_BLOCKS = {
    "left": ("left_move_fast", "turn_left"),
    "right": ("right_move_fast", "turn_right"),
}

# Next-step boxes in the pelvis frame, metres. The near edge stays past the
# toes so the robot's own feet are not "clutter". Edit here, not at the call.
FOOTPRINTS: dict[str, tuple[float, float, float, float]] = {
    # name: (x0, x1, y0, y1)
    # Near edge is past the feet. Closer than this is the robot, and a look
    # that includes it vetoes every step into back_fast.
    "go_forward": (0.28, 0.55, -0.11, 0.11),
    "back_fast": (-0.28, -0.08, -0.09, 0.09),
    "left_move_fast": (0.28, 0.60, 0.08, 0.32),
    "right_move_fast": (0.28, 0.60, -0.32, -0.08),
    "turn_left": (0.26, 0.55, -0.06, 0.30),
    "turn_right": (0.26, 0.55, -0.30, 0.06),
}

FORWARD_CORRIDOR = (0.10, 0.80, -0.12, 0.12)  # range query, wider than the step

# Fraction of footprint samples that must be clutter before a veto.
# A wire is thin; 1.2% of a stride-2 grid is already a real line, not speckle.
CLUTTER_FRACTION = 0.012
CLUTTER_MIN_COUNT = 8
# If semantic occupancy fills the footprint, the floor was labeled "obstacle".
# Drop semantic for that box and keep texture clutter only. Walls are already
# handled by the level-view policy and the ToF beam.
SEMANTIC_SATURATION = 0.35

MOTION_ACTIONS = frozenset(FOOTPRINTS)
TRANSLATIONS = frozenset({"go_forward", "back_fast", "left_move_fast", "right_move_fast"})
# Fallback intrinsics: goal.py's 640x480 pinhole (fx=425). Scaled to the frame.
FALLBACK_FX = 425.0
FALLBACK_SIZE = (640, 480)


def aim_pulses(
    x: float,
    y: float,
    pitch_sign: int = -1,
    yaw_sign: int = 1,
    center: int = CENTER_PULSE,
) -> tuple[int, int]:
    """Pitch and yaw pulses that put the optical axis on a floor point.

    `x`, `y` are pelvis-frame metres. Pitch pulse is on the down side of
    `center` according to `pitch_sign`; yaw pulse is on the left side according
    to `yaw_sign` (same convention as `survey`).
    """
    dx = float(x) - float(CAM_POS_BODY[0])
    dy = float(y) - float(CAM_POS_BODY[1])
    dz = float(GROUND_Z) - float(CAM_POS_BODY[2])
    horiz = max(math.hypot(dx, dy), 1e-3)
    pitch_down_deg = math.degrees(math.atan2(-dz, horiz))
    yaw_deg = math.degrees(math.atan2(dy, dx))
    pitch_pulse = int(round(center + int(pitch_sign) * (pitch_down_deg / DEG_PER_PULSE)))
    yaw_pulse = int(round(center + int(yaw_sign) * (yaw_deg / DEG_PER_PULSE)))
    return (
        int(np.clip(pitch_pulse, PULSE_MIN, PULSE_MAX)),
        int(np.clip(yaw_pulse, PULSE_MIN, PULSE_MAX)),
    )


def pitch_down_rad(pitch_pulse: int, center_pulse: int = CENTER_PULSE, pitch_sign: int = -1) -> float:
    """Radians the optical axis is tilted below horizontal. Negative = looking up.

    `pitch_sign` -1 (default, hand_nod.py): a pulse below 1500 looks down.
    Flip to +1 only if this robot's servo 1 is mounted the other way, and
    put the look pulse on the down side of 1500 to match.
    """
    deg = int(pitch_sign) * (int(pitch_pulse) - int(center_pulse)) * DEG_PER_PULSE
    return math.radians(deg)


def intrinsics_for(width: int, height: int, mtx: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy for a frame. `mtx` is the 640x480 calibration, scaled."""
    if mtx is not None:
        m = np.asarray(mtx, dtype=np.float64)
        sx = width / FALLBACK_SIZE[0]
        sy = height / FALLBACK_SIZE[1]
        return float(m[0, 0] * sx), float(m[1, 1] * sy), float(m[0, 2] * sx), float(m[1, 2] * sy)
    sx = width / FALLBACK_SIZE[0]
    sy = height / FALLBACK_SIZE[1]
    return FALLBACK_FX * sx, FALLBACK_FX * sy, 0.5 * width, 0.5 * height


def ground_points(
    us: np.ndarray,
    vs: np.ndarray,
    pitch_down: float,
    yaw: float,
    K: tuple[float, float, float, float],
    cam_pos: np.ndarray = CAM_POS_BODY,
    ground_z: float = GROUND_Z,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project pixels onto the floor.

    Returns x, y in the pelvis frame and a bool mask of rays that hit the floor
    in front of the camera. Pitch is about body +Y (positive pitches the
    optical axis down). Yaw is about body +Z (positive turns the head left).
    """
    fx, fy, cx, cy = K
    u = np.asarray(us, dtype=np.float64)
    v = np.asarray(vs, dtype=np.float64)
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    # OpenCV ray → body at pitch 0 (goal.R_BODY_FROM_CAM): forward, left, up.
    bx = np.ones_like(x_c)
    by = -x_c
    bz = -y_c
    ca, sa = math.cos(pitch_down), math.sin(pitch_down)
    # Ry(+pitch_down) takes the forward ray downward (negative body Z).
    bx2 = bx * ca + bz * sa
    bz2 = -bx * sa + bz * ca
    by2 = by
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    bx3 = cyaw * bx2 - syaw * by2
    by3 = syaw * bx2 + cyaw * by2
    bz3 = bz2
    denom_ok = bz3 < -1e-4
    t = np.zeros_like(bx3)
    t[denom_ok] = (ground_z - cam_pos[2]) / bz3[denom_ok]
    valid = denom_ok & (t > 0.0)
    x = cam_pos[0] + t * bx3
    y = cam_pos[1] + t * by3
    # Own feet / behind the toes: not a place we can see a wire we haven't
    # already stepped on, and the shin lives here when the head is down.
    valid &= x > 0.08
    return x, y, valid


def _in_box(x: np.ndarray, y: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = box
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def clutter_mask(bgr: np.ndarray) -> np.ndarray:
    """Bool mask of pixels that are not the local floor: thin dark lines and blobs.

    Black-hat catches wires (thinner than the 9px kernel). A colour deviation
    from the median of the central band catches small blocks whose class the
    segmenter called "floor". Textureless floor returns an empty mask.
    """
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thin = blackhat > 18
    h, w = gray.shape
    roi = bgr[int(h * 0.40) : int(h * 0.80), int(w * 0.30) : int(w * 0.70)]
    if roi.size == 0:
        med = np.array([0.0, 0.0, 0.0])
    else:
        med = np.median(roi.reshape(-1, 3), axis=0)
    diff = np.linalg.norm(bgr.astype(np.float32) - med.reshape(1, 1, 3), axis=2)
    blob = diff > 45.0
    mask = (thin | blob).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask > 0


def shift_to_motion(
    shift_xy: tuple[float, float],
    pitch_down: float,
    K: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Metres the robot moved, from a pixel shift of the look-down image.

    `shift_xy` is OpenCV phaseCorrelate(prev, curr): positive x means the
    texture moved right in the image, positive y means it moved down.
    Motion = ground(center) - ground(center + shift), so texture moving down
    (closer pixel) is a forward step.
    """
    fx, fy, cx, cy = K
    sx, sy = float(shift_xy[0]), float(shift_xy[1])
    if abs(sx) > width * 0.25 or abs(sy) > height * 0.25:
        return None
    us = np.array([cx, cx + sx])
    vs = np.array([cy, cy + sy])
    xs, ys, valid = ground_points(us, vs, pitch_down, 0.0, K)
    if not bool(valid.all()):
        return None
    return float(xs[0] - xs[1]), float(ys[0] - ys[1])


def floor_shift_m(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pitch_down: float,
    K: tuple[float, float, float, float],
) -> tuple[float, float, float] | None:
    """(dx_body, dy_body, response) or None when the floor has no texture.

    Response is OpenCV's phase-correlation peak in [0, 1]. Callers should
    ignore results below ~0.15.
    """
    import cv2

    if prev_bgr.shape != curr_bgr.shape:
        return None
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape
    y0, y1 = int(h * 0.40), int(h * 0.75)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    a = g0[y0:y1, x0:x1].astype(np.float32)
    b = g1[y0:y1, x0:x1].astype(np.float32)
    if a.size < 64 or float(a.std()) < 3.0 or float(b.std()) < 3.0:
        return None
    a = a - float(a.mean())
    b = b - float(b.mean())
    (sx, sy), response = cv2.phaseCorrelate(a, b)
    motion = shift_to_motion((sx, sy), pitch_down, K, w, h)
    if motion is None:
        return None
    return motion[0], motion[1], float(response)


def fused_tof(tof_m: float, forward_range_m: float | None, tof_cutoff: float = 1.2) -> float:
    """Range the policy should see: the closer of the chest beam and the floor look.

    No floor detection → the beam is unchanged. A detection never reports past
    the cutoff the policy was trained with, and never below 8 cm (a single
    noisy pixel must not look like a collision).
    """
    if forward_range_m is None or not math.isfinite(forward_range_m):
        return float(tof_m)
    if forward_range_m <= 0.0:
        return float(tof_m)
    return float(min(tof_m, max(0.08, min(forward_range_m, tof_cutoff))))


def avoid_action(
    proposed: str,
    blocked: dict[str, bool],
    clearer_side: str,
) -> tuple[str, str | None]:
    """Replace `proposed` when its footprint is blocked. None reason = keep it.

    `clearer_side` is "left" or "right" (which yaw view had less clutter).
    """
    if proposed not in MOTION_ACTIONS or not blocked.get(proposed, False):
        return proposed, None
    left_turn = "turn_left" if clearer_side != "right" else "turn_right"
    right_turn = "turn_right" if left_turn == "turn_left" else "turn_left"
    order = {
        "go_forward": [left_turn, right_turn, "back_fast", "stand"],
        "left_move_fast": ["turn_right", "back_fast", "go_forward", "stand"],
        "right_move_fast": ["turn_left", "back_fast", "go_forward", "stand"],
        "turn_left": ["turn_right", "back_fast", "stand"],
        "turn_right": ["turn_left", "back_fast", "stand"],
        "back_fast": ["turn_left", "turn_right", "stand"],
    }
    for cand in order.get(proposed, ["stand"]):
        if cand == "stand" or not blocked.get(cand, False):
            return cand, "ground_look"
    return "stand", "ground_look"


@dataclass
class LookResult:
    """One pre-step survey. `aborted` means e-stop fired mid-look."""

    aborted: bool = False
    forward_range_m: float | None = None
    odom_dx: float | None = None
    odom_dy: float | None = None
    odom_response: float = 0.0
    blocked: dict[str, bool] = field(default_factory=dict)
    fractions: dict[str, float] = field(default_factory=dict)
    clearer_side: str = "left"
    semantic_saturated: bool = False
    poses: list[str] = field(default_factory=list)

    def avoid(self, proposed: str) -> tuple[str, str | None]:
        return avoid_action(proposed, self.blocked, self.clearer_side)

    def log_fields(self) -> dict:
        return {
            "ground_range_m": None if self.forward_range_m is None else round(self.forward_range_m, 3),
            "ground_blocked": {k: bool(v) for k, v in self.blocked.items() if v},
            "ground_frac": {k: round(float(v), 4) for k, v in self.fractions.items()},
            "odom_dx": None if self.odom_dx is None else round(self.odom_dx, 4),
            "odom_dy": None if self.odom_dy is None else round(self.odom_dy, 4),
            "odom_response": round(self.odom_response, 3),
            "look_poses": list(self.poses),
            "clearer_side": self.clearer_side,
        }


class GroundLook:
    """Moves the head, reads frames, recenters. Safe to construct on the laptop."""

    def __init__(
        self,
        enabled: bool = True,
        look_pulse: int = DEFAULT_LOOK_PULSE,
        sweep: bool = True,
        yaw_amp: int = DEFAULT_YAW_AMP,
        yaw_sign: int = 1,
        pitch_sign: int = -1,
        center_pulse: int = CENTER_PULSE,
        mtx: np.ndarray | None = None,
        min_odom_response: float = 0.15,
    ):
        self.enabled = bool(enabled)
        self.look_pulse = int(look_pulse)
        self.sweep = bool(sweep)
        self.yaw_amp = int(yaw_amp)
        self.yaw_sign = 1 if int(yaw_sign) >= 0 else -1
        self.pitch_sign = -1 if int(pitch_sign) <= 0 else 1
        self.center_pulse = int(center_pulse)
        self.mtx = mtx
        self.min_odom_response = float(min_odom_response)
        self._prev_bgr: np.ndarray | None = None
        self._prev_action: str | None = None
        self._cmd_pitch = self.center_pulse
        self._cmd_yaw = self.center_pulse

    def note_executed(self, action_name: str) -> None:
        """Call with the clip that was actually played, so the next survey can measure it.

        A stand or a turn breaks the floor-shift chain: the next translation is
        measured against a fresh frame, and the primitive is left as-is.
        """
        self._prev_action = action_name
        if action_name not in TRANSLATIONS:
            self._prev_bgr = None

    def _poses_for(self, action_name: str) -> list[tuple[str, int, int]]:
        """Forward look, a look at each foot, then a head turn to each side.

        Foot poses pitch steeply so the optical axis is the strafe step. Head
        poses yaw much further (~75°) and only about 45° down, so the camera
        sees the floor beside the body rather than the toes. Order keeps the
        long yaw swing to one crossing. `back_fast` takes them too: a blocked
        retreat turns into that same side wire.
        """
        del action_name  # every motion gets the same looks
        pitch = int(np.clip(self.look_pulse, PULSE_MIN, PULSE_MAX))
        center = ("center", pitch, self.center_pulse)
        if not self.sweep:
            return [center]
        order = ("head_left", "left", "right", "head_right")
        aims = {**HEAD_PAN_AIM_M, **SIDE_AIM_M}
        poses = [center]
        for label in order:
            x, y = aims[label]
            pp, yp = aim_pulses(x, y, self.pitch_sign, self.yaw_sign, self.center_pulse)
            poses.append((label, pp, yp))
        return poses

    def _move_ms(self, pitch_pulse: int, yaw_pulse: int) -> int:
        """Servo time for this move. A side look is ~700 µs; 400 ms does not finish it."""
        delta = max(abs(int(pitch_pulse) - self._cmd_pitch), abs(int(yaw_pulse) - self._cmd_yaw))
        return int(np.clip(delta * 1.2, 350, 1000))

    def _settle(self, hw, seconds: float) -> None:
        if getattr(hw, "instant_head", False) or seconds <= 0.0:
            return
        time.sleep(seconds)

    def survey(self, hw, encode_views, action_name: str) -> LookResult:
        """Look down, then put the head back. `encode_views(bgr)` → dict with full_mask.

        `full_mask` is optional; texture clutter is enough for wires. The head is
        recentered even when a pose fails.
        """
        result = LookResult()
        if not self.enabled or action_name not in MOTION_ACTIONS:
            return result
        pitch = pitch_down_rad(self.look_pulse, self.center_pulse, self.pitch_sign)
        if pitch < math.radians(20.0):
            logger.warning(
                "look pulse %d is not below center %d; head would not see the floor. Skipping veto.",
                self.look_pulse,
                self.center_pulse,
            )
            return result
        setter = getattr(hw, "set_head", None)
        if not callable(setter):
            logger.warning("hardware has no set_head; ground look disabled for this run")
            self.enabled = False
            return result

        # Samples accumulated in the body frame across poses.
        hits_x: list[np.ndarray] = []
        hits_y: list[np.ndarray] = []
        clutter_x: list[np.ndarray] = []
        clutter_y: list[np.ndarray] = []
        semantic_x: list[np.ndarray] = []
        semantic_y: list[np.ndarray] = []
        center_bgr = None
        try:
            for label, pitch_pulse, yaw_pulse in self._poses_for(action_name):
                if hw.poll_estop():
                    result.aborted = True
                    return result
                move_ms = self._move_ms(pitch_pulse, yaw_pulse)
                moved = setter(pitch_pulse, yaw_pulse, move_ms)
                if moved is False:
                    logger.warning("set_head failed; not vetoing from a level frame")
                    return LookResult()
                self._cmd_pitch, self._cmd_yaw = int(pitch_pulse), int(yaw_pulse)
                # aim_pulses already applied yaw_sign to the pulse. Multiplying the
                # pulse delta by yaw_sign again makes the "left" pose +yaw (head
                # left) whichever way the servo is mounted.
                yaw = math.radians((yaw_pulse - self.center_pulse) * DEG_PER_PULSE) * self.yaw_sign
                pose_pitch = pitch_down_rad(pitch_pulse, self.center_pulse, self.pitch_sign)
                self._settle(hw, move_ms / 1000.0 + 0.12)
                bgr = hw.read_bgr()
                self._settle(hw, 0.05)
                bgr2 = hw.read_bgr()
                if bgr2 is not None:
                    bgr = bgr2
                if bgr is None:
                    logger.warning("ground look %s: no frame", label)
                    continue
                result.poses.append(label)
                if label == "center":
                    center_bgr = bgr
                # A hard yaw puts the shoulder in the lower corners. Drop that
                # border on the head-turn frames only, so the arm is not a wire.
                self._accumulate(
                    bgr,
                    encode_views,
                    pose_pitch,
                    yaw,
                    hits_x,
                    hits_y,
                    clutter_x,
                    clutter_y,
                    semantic_x,
                    semantic_y,
                    crop_border=0.10 if label.startswith("head_") else 0.0,
                )
        finally:
            try:
                back_ms = self._move_ms(self.center_pulse, self.center_pulse)
                setter(self.center_pulse, self.center_pulse, back_ms)
                self._cmd_pitch, self._cmd_yaw = self.center_pulse, self.center_pulse
                self._settle(hw, back_ms / 1000.0 + 0.08)
            except Exception as e:
                logger.warning("center_head after ground look failed: %s", e)

        if center_bgr is not None and self._prev_bgr is not None and self._prev_action in TRANSLATIONS:
            h, w = center_bgr.shape[:2]
            K = intrinsics_for(w, h, self.mtx)
            shift = floor_shift_m(self._prev_bgr, center_bgr, pitch, K)
            if shift is not None and shift[2] >= self.min_odom_response:
                dx, dy, resp = shift
                if abs(dx) <= 0.20 and abs(dy) <= 0.15:
                    result.odom_dx, result.odom_dy, result.odom_response = dx, dy, resp
        if center_bgr is not None:
            self._prev_bgr = center_bgr

        self._summarize(result, hits_x, hits_y, clutter_x, clutter_y, semantic_x, semantic_y)
        return result

    def _accumulate(
        self,
        bgr,
        encode_views,
        pitch,
        yaw,
        hits_x,
        hits_y,
        clutter_x,
        clutter_y,
        semantic_x,
        semantic_y,
        crop_border: float = 0.0,
    ):
        h, w = bgr.shape[:2]
        K = intrinsics_for(w, h, self.mtx)
        stride = 2
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        us = us.ravel()
        vs = vs.ravel()
        xs, ys, valid = ground_points(us, vs, pitch, yaw, K)
        if crop_border > 0.0:
            m = float(crop_border)
            valid &= (us > w * m) & (us < w * (1.0 - m)) & (vs > h * m) & (vs < h * (1.0 - m))
        if not np.any(valid):
            return
        texture = clutter_mask(bgr)
        sem = np.zeros((h, w), dtype=bool)
        if encode_views is not None:
            try:
                views = encode_views(bgr)
                full = None if views is None else views.get("full_mask")
                if full is not None and full.shape[:2] == (h, w):
                    sem = full > 0
            except Exception as e:
                logger.warning("ground-look segmentation failed: %s", e)
        ui = us[valid].astype(np.int32)
        vi = vs[valid].astype(np.int32)
        hits_x.append(xs[valid])
        hits_y.append(ys[valid])
        tex_hit = texture[vi, ui]
        sem_hit = sem[vi, ui]
        if np.any(tex_hit):
            clutter_x.append(xs[valid][tex_hit])
            clutter_y.append(ys[valid][tex_hit])
        if np.any(sem_hit):
            semantic_x.append(xs[valid][sem_hit])
            semantic_y.append(ys[valid][sem_hit])

    def _summarize(self, result, hits_x, hits_y, clutter_x, clutter_y, semantic_x, semantic_y):
        if not hits_x:
            return
        hx = np.concatenate(hits_x)
        hy = np.concatenate(hits_y)
        tx = np.concatenate(clutter_x) if clutter_x else np.zeros(0)
        ty = np.concatenate(clutter_y) if clutter_y else np.zeros(0)
        sx = np.concatenate(semantic_x) if semantic_x else np.zeros(0)
        sy = np.concatenate(semantic_y) if semantic_y else np.zeros(0)

        for name, box in FOOTPRINTS.items():
            in_box = _in_box(hx, hy, box)
            n = int(in_box.sum())
            if n == 0:
                result.blocked[name] = False
                result.fractions[name] = 0.0
                continue
            tex_n = int(_in_box(tx, ty, box).sum()) if tx.size else 0
            sem_n = int(_in_box(sx, sy, box).sum()) if sx.size else 0
            sem_frac = sem_n / float(n)
            use_sem = sem_frac < SEMANTIC_SATURATION
            if not use_sem:
                result.semantic_saturated = True
            count = tex_n + (sem_n if use_sem else 0)
            # Texture and semantic can mark the same pixel; the cap keeps the
            # fraction ≤ 1 when both fire.
            count = min(count, n)
            frac = count / float(n)
            result.fractions[name] = frac
            result.blocked[name] = frac >= CLUTTER_FRACTION and count >= CLUTTER_MIN_COUNT

        # A wire beside the body is not in the forward step. It still blocks a
        # strafe or a turn toward that side. Straight ahead stays free.
        for side, box in SIDE_CORRIDORS.items():
            in_box = _in_box(hx, hy, box)
            n = int(in_box.sum())
            if n == 0:
                continue
            tex_n = int(_in_box(tx, ty, box).sum()) if tx.size else 0
            sem_n = int(_in_box(sx, sy, box).sum()) if sx.size else 0
            use_sem = (sem_n / float(n)) < SEMANTIC_SATURATION
            count = min(n, tex_n + (sem_n if use_sem else 0))
            if count >= CLUTTER_MIN_COUNT and count / float(n) >= CLUTTER_FRACTION:
                for name in SIDE_BLOCKS[side]:
                    result.blocked[name] = True
                    result.fractions[name] = max(result.fractions.get(name, 0.0), count / float(n))

        # Range uses texture always, and semantic only when it does not fill the
        # corridor (a filled corridor is a mislabeled floor, not an object).
        pieces = []
        if tx.size:
            pieces.append(tx[_in_box(tx, ty, FORWARD_CORRIDOR)])
        corridor_hits = int(_in_box(hx, hy, FORWARD_CORRIDOR).sum())
        if sx.size and corridor_hits > 0:
            sem_in = _in_box(sx, sy, FORWARD_CORRIDOR)
            if int(sem_in.sum()) / float(corridor_hits) < SEMANTIC_SATURATION:
                pieces.append(sx[sem_in])
        ranges = np.concatenate(pieces) if pieces else np.zeros(0)
        if ranges.size >= CLUTTER_MIN_COUNT:
            result.forward_range_m = float(np.percentile(ranges, 10))

        left_box = (0.05, 0.55, 0.05, 0.55)
        right_box = (0.05, 0.55, -0.55, -0.05)
        left_n = int(_in_box(tx, ty, left_box).sum()) if tx.size else 0
        right_n = int(_in_box(tx, ty, right_box).sum()) if tx.size else 0
        result.clearer_side = "left" if left_n <= right_n else "right"
