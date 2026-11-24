"""World-frame occupancy grid: Dijkstra cost-to-go + A* polyline (nav expert).

Privileged (train-only): GT axis-aligned boxes, not the camera 64x64 mask.
No mujoco import — spawn workers can import this after the env does.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)
# 8-connected: (di, dj, cost_in_cells, is_diag, ortho_a, ortho_b)
_NEIGH = (
    (-1, 0, 1.0, False, None, None),
    (1, 0, 1.0, False, None, None),
    (0, -1, 1.0, False, None, None),
    (0, 1, 1.0, False, None, None),
    (-1, -1, SQRT2, True, (-1, 0), (0, -1)),
    (-1, 1, SQRT2, True, (-1, 0), (0, 1)),
    (1, -1, SQRT2, True, (1, 0), (0, -1)),
    (1, 1, SQRT2, True, (1, 0), (0, 1)),
)


class OccupancyGrid2D:
    """Binary occupancy + metres-to-goal field on a uniform XY grid."""

    def __init__(
        self,
        origin_xy: tuple[float, float] = (-3.2, -3.2),
        size_xy: tuple[float, float] = (6.4, 6.4),
        resolution: float = 0.05,
        robot_radius: float = 0.18,
    ):
        self.origin_x = float(origin_xy[0])
        self.origin_y = float(origin_xy[1])
        self.res = float(resolution)
        self.robot_radius = float(robot_radius)
        self.w = int(round(float(size_xy[0]) / self.res))
        self.h = int(round(float(size_xy[1]) / self.res))
        if self.w < 4 or self.h < 4:
            raise ValueError(f"grid too small: {self.w}x{self.h}")
        self.occ = np.zeros((self.h, self.w), dtype=bool)
        self.cost = np.full((self.h, self.w), np.inf, dtype=np.float64)
        self.goal_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.reachable = False

    def clear(self) -> None:
        self.occ[:] = False
        self.cost[:] = np.inf
        self.reachable = False

    def in_bounds_ij(self, i: int, j: int) -> bool:
        return 0 <= i < self.h and 0 <= j < self.w

    def world_to_ij(self, x: float, y: float) -> tuple[int, int]:
        j = int(math.floor((float(x) - self.origin_x) / self.res))
        i = int(math.floor((float(y) - self.origin_y) / self.res))
        return i, j

    def ij_to_world(self, i: int, j: int) -> tuple[float, float]:
        x = self.origin_x + (j + 0.5) * self.res
        y = self.origin_y + (i + 0.5) * self.res
        return x, y

    def in_collision_xy(self, x: float, y: float) -> bool:
        i, j = self.world_to_ij(x, y)
        if not self.in_bounds_ij(i, j):
            return True
        return bool(self.occ[i, j])

    def stamp_box(self, cx: float, cy: float, hx: float, hy: float) -> None:
        """Mark cells whose center is within robot_radius of the AABB (disk Minkowski)."""
        r = self.robot_radius
        x0 = cx - hx - r
        x1 = cx + hx + r
        y0 = cy - hy - r
        y1 = cy + hy + r
        j0 = max(0, int(math.floor((x0 - self.origin_x) / self.res)))
        j1 = min(self.w - 1, int(math.floor((x1 - self.origin_x) / self.res)))
        i0 = max(0, int(math.floor((y0 - self.origin_y) / self.res)))
        i1 = min(self.h - 1, int(math.floor((y1 - self.origin_y) / self.res)))
        if j1 < j0 or i1 < i0:
            return
        js = np.arange(j0, j1 + 1, dtype=np.float64)
        is_ = np.arange(i0, i1 + 1, dtype=np.float64)
        xs = self.origin_x + (js + 0.5) * self.res
        ys = self.origin_y + (is_ + 0.5) * self.res
        xx, yy = np.meshgrid(xs, ys)
        dx = np.maximum(0.0, np.abs(xx - cx) - hx)
        dy = np.maximum(0.0, np.abs(yy - cy) - hy)
        self.occ[i0 : i1 + 1, j0 : j1 + 1] |= (dx * dx + dy * dy) <= (r * r)

    def clear_disk(self, x: float, y: float, radius: float) -> None:
        r = float(radius)
        j0 = max(0, int(math.floor((x - r - self.origin_x) / self.res)))
        j1 = min(self.w - 1, int(math.floor((x + r - self.origin_x) / self.res)))
        i0 = max(0, int(math.floor((y - r - self.origin_y) / self.res)))
        i1 = min(self.h - 1, int(math.floor((y + r - self.origin_y) / self.res)))
        if j1 < j0 or i1 < i0:
            return
        js = np.arange(j0, j1 + 1, dtype=np.float64)
        is_ = np.arange(i0, i1 + 1, dtype=np.float64)
        xs = self.origin_x + (js + 0.5) * self.res
        ys = self.origin_y + (is_ + 0.5) * self.res
        xx, yy = np.meshgrid(xs, ys)
        self.occ[i0 : i1 + 1, j0 : j1 + 1] &= ((xx - x) ** 2 + (yy - y) ** 2) > (r * r)

    def nearest_free(self, x: float, y: float, max_r_cells: int = 8) -> tuple[int, int] | None:
        si, sj = self.world_to_ij(x, y)
        if self.in_bounds_ij(si, sj) and not self.occ[si, sj]:
            return si, sj
        for rad in range(1, max_r_cells + 1):
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    if max(abs(di), abs(dj)) != rad:
                        continue
                    i, j = si + di, sj + dj
                    if self.in_bounds_ij(i, j) and not self.occ[i, j]:
                        return i, j
        return None

    def dijkstra(self, goal_xy) -> None:
        """Cost-to-go in metres from every free cell to goal. inf if unreachable."""
        self.cost[:] = np.inf
        self.reachable = False
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        seed = self.nearest_free(float(self.goal_xy[0]), float(self.goal_xy[1]))
        if seed is None:
            return
        gi, gj = seed
        self.cost[gi, gj] = 0.0
        heap: list[tuple[float, int, int]] = [(0.0, gi, gj)]
        occ = self.occ
        cost = self.cost
        h, w = self.h, self.w
        res = self.res
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > cost[i, j]:
                continue
            for di, dj, ccells, is_diag, oa, ob in _NEIGH:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= h or nj < 0 or nj >= w:
                    continue
                if occ[ni, nj]:
                    continue
                if is_diag:
                    ai, aj = i + oa[0], j + oa[1]
                    bi, bj = i + ob[0], j + ob[1]
                    if occ[ai, aj] or occ[bi, bj]:
                        continue
                nd = d + ccells * res
                if nd < cost[ni, nj]:
                    cost[ni, nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))
        self.reachable = bool(np.isfinite(self.cost).any())

    def geodesic(self, x: float, y: float) -> float:
        i, j = self.world_to_ij(x, y)
        if not self.in_bounds_ij(i, j):
            return float("inf")
        return float(self.cost[i, j])

    def descent_yaw(self, x: float, y: float) -> float | None:
        """Yaw of the cost-to-go descent direction, or None if undefined."""
        i, j = self.world_to_ij(x, y)
        if not self.in_bounds_ij(i, j) or not np.isfinite(self.cost[i, j]):
            return None
        best = None
        best_c = self.cost[i, j]
        for di, dj, _, is_diag, oa, ob in _NEIGH:
            ni, nj = i + di, j + dj
            if not self.in_bounds_ij(ni, nj) or self.occ[ni, nj]:
                continue
            if is_diag:
                if self.occ[i + oa[0], j + oa[1]] or self.occ[i + ob[0], j + ob[1]]:
                    continue
            c = self.cost[ni, nj]
            if np.isfinite(c) and c < best_c - 1e-9:
                best_c = c
                wx, wy = self.ij_to_world(ni, nj)
                best = math.atan2(wy - y, wx - x)
        return best

    def astar(self, start_xy, goal_xy=None, max_expands: int = 20000) -> np.ndarray | None:
        """8-connected A* world-XY polyline (no corner cutting). None if unreachable.

        If Dijkstra has already filled `cost`, that field is an exact heuristic so
        A* follows the geodesic with almost no extra expansions.
        """
        if goal_xy is None:
            gx, gy = float(self.goal_xy[0]), float(self.goal_xy[1])
        else:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
        start = self.nearest_free(float(start_xy[0]), float(start_xy[1]))
        goal = self.nearest_free(gx, gy)
        if start is None or goal is None:
            return None
        si, sj = start
        gi, gj = goal
        start_w = self.ij_to_world(si, sj)
        if (si, sj) == (gi, gj):
            if abs(start_w[0] - gx) < 1e-9 and abs(start_w[1] - gy) < 1e-9:
                return np.asarray([start_w], dtype=np.float64)
            return np.asarray([start_w, (gx, gy)], dtype=np.float64)

        occ = self.occ
        cost = self.cost
        h, w = self.h, self.w
        res = self.res

        def heuristic(i: int, j: int) -> float:
            c = cost[i, j]
            if np.isfinite(c):
                return float(c)
            return math.hypot(i - gi, j - gj) * res

        gscore = {(si, sj): 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        closed: set[tuple[int, int]] = set()
        counter = 0
        heap: list[tuple[float, float, int, int, int]] = [
            (heuristic(si, sj), 0.0, counter, si, sj)
        ]
        found = False
        expands = 0
        while heap:
            _f, g, _, i, j = heapq.heappop(heap)
            if (i, j) in closed:
                continue
            closed.add((i, j))
            expands += 1
            if (i, j) == (gi, gj):
                found = True
                break
            if expands >= max_expands:
                break
            for di, dj, ccells, is_diag, oa, ob in _NEIGH:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= h or nj < 0 or nj >= w or occ[ni, nj]:
                    continue
                if is_diag:
                    if occ[i + oa[0], j + oa[1]] or occ[i + ob[0], j + ob[1]]:
                        continue
                ng = g + ccells * res
                key = (ni, nj)
                if ng + 1e-12 >= gscore.get(key, 1e100):
                    continue
                gscore[key] = ng
                came[key] = (i, j)
                counter += 1
                heapq.heappush(heap, (ng + heuristic(ni, nj), ng, counter, ni, nj))

        if not found:
            return None
        path_ij = [(gi, gj)]
        cur = (gi, gj)
        while cur != (si, sj):
            cur = came.get(cur)
            if cur is None:
                return None
            path_ij.append(cur)
        path_ij.reverse()
        pts = [self.ij_to_world(i, j) for i, j in path_ij]
        if abs(pts[-1][0] - gx) > 1e-6 or abs(pts[-1][1] - gy) > 1e-6:
            pts.append((gx, gy))
        return np.asarray(pts, dtype=np.float64)

    def polyline(self, start_xy, max_steps: int = 400) -> np.ndarray | None:
        """A* path from start to the goal seed. None if unreachable."""
        return self.astar(start_xy, max_expands=max(20000, int(max_steps) * 50))


def build_grid(
    boxes,
    goal_xy,
    clear_xy=(),
    origin_xy=(-3.2, -3.2),
    size_xy=(6.4, 6.4),
    resolution=0.05,
    robot_radius=0.18,
    clear_radius=0.10,
) -> OccupancyGrid2D:
    grid = OccupancyGrid2D(
        origin_xy=origin_xy,
        size_xy=size_xy,
        resolution=resolution,
        robot_radius=robot_radius,
    )
    for box in boxes:
        grid.stamp_box(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    grid.clear_disk(gx, gy, clear_radius)
    for xy in clear_xy:
        grid.clear_disk(float(xy[0]), float(xy[1]), clear_radius)
    grid.dijkstra((gx, gy))
    return grid


if __name__ == "__main__":
    # Demo layout from TonyPiNavEnv (obs half-extents from scene_nav.xml).
    boxes = [
        (0.95, 0.02, 0.09, 0.09),
        (1.35, -0.48, 0.10, 0.08),
        (1.60, 0.42, 0.08, 0.11),
        (0.72, -0.36, 0.12, 0.07),
    ]
    goal = (2.0, -0.5)
    grid = build_grid(boxes, goal, clear_xy=((0.0, 0.0),))
    g0 = grid.geodesic(0.0, 0.0)
    path = grid.astar((0.0, 0.0))
    eucl = math.hypot(goal[0], goal[1])
    print(f"reachable={grid.reachable} geodesic={g0:.3f}m euclid={eucl:.3f}m")
    if path is None:
        raise SystemExit("planner self-check failed: no A* path")
    plen = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    print(f"astar n={len(path)} length={plen:.3f}m")
    # Euclidean-heuristic A* (wipe the Dijkstra field) must match.
    saved = grid.cost.copy()
    grid.cost[:] = np.inf
    path_eucl = grid.astar((0.0, 0.0))
    grid.cost[:] = saved
    if path_eucl is None:
        raise SystemExit("euclidean-heuristic A* failed")
    if not np.isfinite(g0) or g0 < eucl - 0.05:
        raise SystemExit(f"geodesic {g0} should be >= euclid {eucl} (minus snap)")
    if plen < 2.0 or plen > 5.0:
        raise SystemExit(f"unexpected path length {plen}")
    # Must not punch through the centerline box at x≈0.95, y≈0.
    for p in path:
        if abs(p[0] - 0.95) < 0.12 and abs(p[1] - 0.02) < 0.12:
            raise SystemExit(f"path cuts the first box: {p}")
    print("nav_planner self-check ok")
