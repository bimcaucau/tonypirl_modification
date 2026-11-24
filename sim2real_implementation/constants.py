"""Constants for TonyPi high-level navigation sim-to-real.

Matches tonypi_rl/tonypi_nav_env.py and SIM2REAL_NAV.md specification.
"""
from __future__ import annotations

# Action names order matches tonypi_rl/tonypi_nav_env.py::ACTION_NAMES
ACTION_NAMES = [
    "stand",
    "go_forward",
    "back_fast",
    "left_move_fast",
    "right_move_fast",
    "turn_left",
    "turn_right",
]

# Action lookup helpers
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}

# Observation scaling constants (must match tonypi_rl/tonypi_nav_env.py)
GOAL_DIST_SCALE = 3.0
TOF_CUTOFF = 1.2
SONAR_CUTOFF = 5.0

# Sim primitives (dx_m, dy_m, dyaw_rad) matching tonypi_rl/tonypi_nav_env.py::PRIMITIVES
SIM_PRIMITIVES = {
    "stand": (0.00, 0.00, 0.00),
    "go_forward": (0.08, 0.00, 0.00),
    "back_fast": (-0.06, 0.00, 0.00),
    "left_move_fast": (0.00, 0.05, 0.00),
    "right_move_fast": (0.00, -0.05, 0.00),
    "turn_left": (0.00, 0.00, 0.35),
    "turn_right": (0.00, 0.00, -0.35),
}

# Occupancy grid size
try:
    from tonypi_rl.perception import OCCUPANCY_SIZE
except Exception:
    # tonypi_rl/perception.py defines OCCUPANCY_SIZE = 64
    OCCUPANCY_SIZE = 64

# Semantic occupancy (ADE20K indoor stuff, table legs included).
# Instance-seg fallback: --yolo-weights yolo26n-seg.pt
DEFAULT_YOLO_WEIGHTS = "yolo26n-sem-ade20k.pt"
YOLO26_SEG_FALLBACK = "yolo26n-seg.pt"
YOLO11_SEG_FALLBACK = "yolo11n-seg.pt"

assert OCCUPANCY_SIZE == 64, f"OCCUPANCY_SIZE must be 64, got {OCCUPANCY_SIZE}"
