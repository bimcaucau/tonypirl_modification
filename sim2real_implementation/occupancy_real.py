"""Occupancy encoders for real-world TonyPi navigation.

Supported backends:
  1. yolo: ultralytics YOLO26-sem ADE20K (default yolo26n-sem-ade20k.pt;
     instance fallback yolo26n-seg.pt / yolo11n-seg.pt)
  2. color: HSV in-range mask with morphological filtering and area resize
  3. file: load static (64, 64) or (1, 64, 64) npy array (offline testing)
  4. zeros: all zeros (1, 64, 64) for open-floor smoke tests

`encode()` is the policy contract and MUST return (1, 64, 64) float32 in {0,1}.
`encode_views()` is for the live 3-pane vis script and debug dumps only.
"""
from __future__ import annotations

import logging
import os
import cv2
import numpy as np

from sim2real_implementation.constants import DEFAULT_YOLO_WEIGHTS, OCCUPANCY_SIZE

logger = logging.getLogger(__name__)

# BGR: obstacle red, free/road green
COLOR_OBSTACLE = (40, 40, 220)
COLOR_ROAD = (40, 180, 40)
COLOR_HSV_TINT = (0, 80, 255)


def hsv_in_range(
    hsv: np.ndarray,
    hsv_lo: list[int] | tuple[int, int, int],
    hsv_hi: list[int] | tuple[int, int, int],
) -> np.ndarray:
    """OpenCV inRange with hue wrap when H_lo > H_hi (red wraps 170–179 + 0–H_hi)."""
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    low_wrap = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], dtype=np.uint8), hi)
    high_wrap = cv2.inRange(
        hsv, lo, np.array([179, hi[1], hi[2]], dtype=np.uint8)
    )
    return cv2.bitwise_or(low_wrap, high_wrap)


def color_mask_full(
    bgr: np.ndarray | None,
    hsv_lo: list[int] | tuple[int, int, int] = (0, 80, 80),
    hsv_hi: list[int] | tuple[int, int, int] = (12, 255, 255),
) -> np.ndarray | None:
    """Full-resolution uint8 {0,255} obstacle mask, or None if no frame."""
    if bgr is None:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = hsv_in_range(hsv, hsv_lo, hsv_hi)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def downsample_mask(full_mask: np.ndarray) -> np.ndarray:
    """uint8/float full-res mask -> (1, 64, 64) float32 in {0,1}."""
    small = cv2.resize(
        full_mask.astype(np.float32),
        (OCCUPANCY_SIZE, OCCUPANCY_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    peak = 255.0 if full_mask.dtype == np.uint8 or small.max() > 1.5 else 1.0
    return (small > peak * 0.3).astype(np.float32)[np.newaxis, ...]


def color_occupancy(
    bgr: np.ndarray | None,
    hsv_lo: list[int] | tuple[int, int, int] = (0, 80, 80),
    hsv_hi: list[int] | tuple[int, int, int] = (12, 255, 255),
) -> np.ndarray:
    """Segment color obstacles in HSV and downsample to (1, 64, 64) float32 in {0.0, 1.0}."""
    full = color_mask_full(bgr, hsv_lo, hsv_hi)
    if full is None:
        logger.debug("bgr frame is None; returning zero occupancy")
        return np.zeros((1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32)
    return downsample_mask(full)


def overlay_mask(bgr: np.ndarray, full_mask: np.ndarray, tint_bgr=COLOR_HSV_TINT, alpha: float = 0.5) -> np.ndarray:
    """Tint pixels where full_mask > 0."""
    out = bgr.copy()
    hit = full_mask > 0
    if not np.any(hit):
        return out
    tint = np.zeros_like(out)
    tint[hit] = tint_bgr
    blended = cv2.addWeighted(out, 1.0 - alpha, tint, alpha, 0)
    out[hit] = blended[hit]
    return out


def binary_obstacle_road(full_mask: np.ndarray, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    """BGR pane: green = road/free, red = obstacle."""
    if shape_hw is None:
        h, w = full_mask.shape[:2]
    else:
        h, w = shape_hw
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = COLOR_ROAD
    if full_mask.shape[0] != h or full_mask.shape[1] != w:
        full_mask = cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    out[full_mask > 0] = COLOR_OBSTACLE
    return out


def empty_views(bgr: np.ndarray | None) -> dict:
    if bgr is None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        canvas = bgr
    h, w = canvas.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    return {
        "occupancy": np.zeros((1, OCCUPANCY_SIZE, OCCUPANCY_SIZE), dtype=np.float32),
        "full_mask": full,
        "overlay": canvas.copy(),
        "binary": binary_obstacle_road(full, (h, w)),
        "camera": canvas,
        "labels": [],
    }


class OccupancyEncoder:
    """Unified occupancy encoder interface for real-time deployment."""

    def __init__(
        self,
        mode: str = "color",
        yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
        yolo_conf: float = 0.25,
        yolo_imgsz: int = 320,
        hsv_lo: list[int] = (0, 80, 80),
        hsv_hi: list[int] = (12, 255, 255),
        file_path: str | None = None,
        keep_all: bool = False,
    ):
        self.mode = mode.lower()
        self.hsv_lo = hsv_lo
        self.hsv_hi = hsv_hi
        self.file_path = file_path
        self.keep_all = keep_all
        self.yolo_weights = yolo_weights
        self._yolo_encoder = None

        if self.mode == "yolo":
            try:
                from tonypi_rl.perception import YoloSegOccupancy
                self._yolo_encoder = YoloSegOccupancy(
                    weights=yolo_weights, conf=yolo_conf, imgsz=yolo_imgsz
                )
                logger.info("Initialized YoloSegOccupancy with weights: %s", yolo_weights)
            except Exception as e:
                logger.error("Failed to initialize YoloSegOccupancy: %s", e)
                raise

    def encode(self, bgr: np.ndarray | None) -> np.ndarray:
        """Produce (1, 64, 64) float32 occupancy grid in {0.0, 1.0}."""
        return self.encode_views(bgr)["occupancy"]

    def encode_views(self, bgr: np.ndarray | None) -> dict:
        """Policy occupancy plus full-res camera / overlay / binary panes."""
        if self.mode == "zeros":
            return empty_views(bgr)

        if self.mode == "file":
            views = empty_views(bgr)
            if self.file_path and os.path.exists(self.file_path):
                try:
                    arr = np.load(self.file_path).astype(np.float32)
                    if arr.ndim == 2 and arr.shape == (OCCUPANCY_SIZE, OCCUPANCY_SIZE):
                        arr = arr[np.newaxis, ...]
                    if arr.shape == (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE):
                        views["occupancy"] = arr
                        h, w = views["camera"].shape[:2]
                        full = cv2.resize(
                            (arr[0] * 255.0).astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        views["full_mask"] = full
                        views["overlay"] = overlay_mask(views["camera"], full)
                        views["binary"] = binary_obstacle_road(full, (h, w))
                except Exception as e:
                    logger.warning("Error loading occupancy file %s: %s", self.file_path, e)
            return views

        if bgr is None:
            return empty_views(None)

        if self.mode == "yolo":
            packed = self._yolo_encoder.encode_views(bgr, keep_all=self.keep_all)
            h, w = bgr.shape[:2]
            packed["camera"] = bgr
            packed["binary"] = binary_obstacle_road(packed["full_mask"], (h, w))
            return packed

        full = color_mask_full(bgr, self.hsv_lo, self.hsv_hi)
        if full is None:
            return empty_views(bgr)
        h, w = bgr.shape[:2]
        return {
            "occupancy": downsample_mask(full),
            "full_mask": full,
            "overlay": overlay_mask(bgr, full),
            "binary": binary_obstacle_road(full, (h, w)),
            "camera": bgr,
            "labels": [("hsv", 1.0, COLOR_HSV_TINT)],
        }
