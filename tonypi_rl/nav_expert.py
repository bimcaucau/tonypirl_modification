"""Receding-horizon Discrete(7) expert: A* trajectory + primitive controller.

Privileged (train-only). Does not import TonyPiNavEnv at module level (cycle).
The env passes ACTION_NAMES / PRIMITIVES in.
"""
from __future__ import annotations

import math

import numpy as np

from tonypi_rl.nav_planner import OccupancyGrid2D, build_grid


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _apply_primitive(x, y, yaw, dx, dy, dyaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x1 = x + c * dx - s * dy
    y1 = y + s * dx + c * dy
    return x1, y1, yaw + dyaw


def _carrot(path: np.ndarray | None, x: float, y: float, look_ahead: float):
    """First path vertex at least `look_ahead` metres ahead of (x, y)."""
    if path is None or len(path) == 0:
        return None
    d2 = (path[:, 0] - x) ** 2 + (path[:, 1] - y) ** 2
    i0 = int(np.argmin(d2))
    acc = 0.0
    px, py = x, y
    for k in range(i0, len(path)):
        qx, qy = float(path[k, 0]), float(path[k, 1])
        acc += math.hypot(qx - px, qy - py)
        px, py = qx, qy
        if acc >= look_ahead:
            return (px, py)
    return (float(path[-1, 0]), float(path[-1, 1]))


class PrimitiveExpert:
    """Plan an A* polyline, then pick the Discrete(7) clip that tracks it."""

    def __init__(
        self,
        action_names,
        primitives: dict,
        robot_radius: float = 0.18,
        grid_res: float = 0.05,
        heading_weight: float = 0.12,
        arrive_radius: float = 0.20,
        look_ahead: float = 0.30,
        origin_xy=(-3.2, -3.2),
        size_xy=(6.4, 6.4),
    ):
        self.action_names = list(action_names)
        self.primitives = primitives
        self.robot_radius = float(robot_radius)
        self.grid_res = float(grid_res)
        self.heading_weight = float(heading_weight)
        self.arrive_radius = float(arrive_radius)
        self.look_ahead = float(look_ahead)
        self.origin_xy = tuple(origin_xy)
        self.size_xy = tuple(size_xy)
        self.grid: OccupancyGrid2D | None = None
        self.goal_xy = np.zeros(2, dtype=np.float64)
        self.unreachable = False
        self._name_to_id = {n: i for i, n in enumerate(self.action_names)}

    def rebuild(self, boxes, goal_xy, clear_xy=()) -> None:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        self.goal_xy = np.array([gx, gy], dtype=np.float64)
        self.grid = build_grid(
            boxes,
            (gx, gy),
            clear_xy=clear_xy,
            origin_xy=self.origin_xy,
            size_xy=self.size_xy,
            resolution=self.grid_res,
            robot_radius=self.robot_radius,
        )
        self.unreachable = not self.grid.reachable

    def geodesic(self, x: float, y: float) -> float:
        if self.grid is None:
            return float("inf")
        g = self.grid.geodesic(x, y)
        if np.isfinite(g):
            return g
        return float("inf")

    def path_polyline(self, x: float, y: float) -> np.ndarray | None:
        if self.grid is None:
            return None
        return self.grid.polyline((x, y))

    def act(self, x: float, y: float, yaw: float) -> int:
        gx, gy = float(self.goal_xy[0]), float(self.goal_xy[1])
        dist = math.hypot(gx - x, gy - y)
        if dist < self.arrive_radius:
            return self._name_to_id.get("stand", 0)

        path = self.path_polyline(x, y)
        carrot = _carrot(path, x, y, self.look_ahead)
        desired = None
        if carrot is not None and math.hypot(carrot[0] - x, carrot[1] - y) > 0.04:
            desired = math.atan2(carrot[1] - y, carrot[0] - x)
        elif self.grid is not None:
            desired = self.grid.descent_yaw(x, y)
        if desired is None:
            desired = math.atan2(gy - y, gx - x)

        best_id = None
        best_score = float("inf")
        geo_now = self.geodesic(x, y)
        use_geo = np.isfinite(geo_now)

        for i, name in enumerate(self.action_names):
            dx, dy, dyaw = self.primitives[name]
            x1, y1, yaw1 = _apply_primitive(x, y, yaw, dx, dy, dyaw)
            colliding = self.grid is not None and self.grid.in_collision_xy(x1, y1)
            if colliding and name != "stand":
                continue
            if use_geo:
                g1 = self.geodesic(x1, y1)
                if not np.isfinite(g1):
                    continue
            else:
                g1 = math.hypot(gx - x1, gy - y1)
            herr = abs(_wrap(yaw1 - desired))
            score = g1 + self.heading_weight * herr
            if name == "stand" and dist >= self.arrive_radius:
                score += 10.0
            elif name == "back_fast":
                score += 0.03
            if score < best_score:
                best_score = score
                best_id = i

        if best_id is not None:
            return int(best_id)
        return self._stuck_action(yaw, desired)

    def _stuck_action(self, yaw: float, desired: float) -> int:
        err = _wrap(desired - yaw)
        if err > 0.05:
            return self._name_to_id.get("turn_left", 5)
        if err < -0.05:
            return self._name_to_id.get("turn_right", 6)
        return self._name_to_id.get("back_fast", 2)
