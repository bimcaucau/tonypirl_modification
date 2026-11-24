"""Occupancy encoders for the high-level nav env.

Training uses a GL-free pinhole projection of obs_* boxes into the head camera
(so PPO workers never create an EGL context — that segfaults in Docker spawn).
Optional MuJoCo segmentation remains for debug. Deploy uses YOLO26-sem
(default yolo26n-sem-ade20k.pt; instance-seg fallback yolo26n-seg.pt).
Both return the same (1, 64, 64) mask.
"""
from __future__ import annotations

import os
import numpy as np

OCCUPANCY_SIZE = 64
SEG_RENDER_SIZE = 64


def _finalize_mask(mask: np.ndarray) -> np.ndarray:
    """(H,W) float -> (1, OCCUPANCY_SIZE, OCCUPANCY_SIZE) binary {0,1}."""
    h, w = mask.shape
    if h == OCCUPANCY_SIZE and w == OCCUPANCY_SIZE:
        occ = mask
    elif h == 2 * OCCUPANCY_SIZE and w == 2 * OCCUPANCY_SIZE:
        occ = mask.reshape(OCCUPANCY_SIZE, 2, OCCUPANCY_SIZE, 2).max(axis=(1, 3))
    else:
        import cv2
        occ = cv2.resize(mask, (OCCUPANCY_SIZE, OCCUPANCY_SIZE), interpolation=cv2.INTER_AREA)
    occ = (occ > 0).astype(np.float32)
    return occ[np.newaxis, ...]

_BOX_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=np.float64,
)


def _fill_hull(mask: np.ndarray, pts: np.ndarray):
    """Fill the convex hull of 2D int points into mask (in-place, values 0/1)."""
    if pts.shape[0] < 3:
        if pts.shape[0] > 0:
            for u, v in pts:
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
                    mask[v, u] = 1.0
        return
    try:
        import cv2
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    except Exception:
        u0, u1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        v0, v1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        u0 = max(0, u0); v0 = max(0, v0)
        u1 = min(mask.shape[1] - 1, u1); v1 = min(mask.shape[0] - 1, v1)
        mask[v0 : v1 + 1, u0 : u1 + 1] = 1.0


class MujocoGtOccupancy:
    """Binary obstacle mask from head_camera.

    Default `use_gl=False`: project obs_* box corners through the pinhole camera.
    `use_gl=True`: MuJoCo segmentation renderer (needs a working MUJOCO_GL).
    """

    def __init__(self, model, camera: str = "head_camera", obstacle_geom_ids=None, use_gl: bool = False):
        import mujoco

        self.model = model
        self.camera = camera
        self.use_gl = bool(use_gl)
        self.obstacle_geom_ids = np.asarray(
            obstacle_geom_ids if obstacle_geom_ids is not None else [],
            dtype=np.int32,
        )
        self.cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera))
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera}' not found in model")
        self.fovy = float(np.deg2rad(model.cam_fovy[self.cam_id]))
        self._geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        self.renderer = None
        self.rgb_renderer = None
        if self.use_gl:
            self._init_gl()

    def _init_gl(self):
        import mujoco
        self.renderer = mujoco.Renderer(self.model, height=SEG_RENDER_SIZE, width=SEG_RENDER_SIZE)
        self.renderer.enable_segmentation_rendering()

    def set_obstacle_ids(self, geom_ids):
        self.obstacle_geom_ids = np.asarray(geom_ids, dtype=np.int32)

    def encode(self, data) -> np.ndarray:
        """Return occupancy (1, 64, 64) float32 in {0, 1}."""
        if self.use_gl:
            return self._encode_gl(data)
        return self._encode_project(data)

    def _encode_gl(self, data) -> np.ndarray:
        if self.renderer is None:
            self._init_gl()
        self.renderer.update_scene(data, camera=self.camera)
        seg = self.renderer.render()
        geom_ids = seg[..., 0]
        obj_type = seg[..., 1]
        if self.obstacle_geom_ids.size == 0:
            mask64 = np.zeros((SEG_RENDER_SIZE, SEG_RENDER_SIZE), dtype=np.float32)
        else:
            is_geom = obj_type == self._geom_type
            is_obs = np.isin(geom_ids, self.obstacle_geom_ids)
            mask64 = (is_geom & is_obs).astype(np.float32)
        return _finalize_mask(mask64)

    def _encode_project(self, data) -> np.ndarray:
        """Pinhole-project each obs_* box into the head camera. No OpenGL."""
        w = h = SEG_RENDER_SIZE
        mask64 = np.zeros((h, w), dtype=np.float32)
        if self.obstacle_geom_ids.size == 0:
            return _finalize_mask(mask64)

        cam_pos = np.asarray(data.cam_xpos[self.cam_id], dtype=np.float64)
        cam_rot = np.asarray(data.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3)
        f = (0.5 * h) / np.tan(0.5 * self.fovy)
        cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)

        for gid in self.obstacle_geom_ids:
            gid = int(gid)
            if gid < 0:
                continue
            size = np.asarray(self.model.geom_size[gid], dtype=np.float64)
            # parked off-arena slots sit around (12, 12)
            xpos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
            if xpos[0] > 8.0:
                continue
            xmat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            corners_w = (xmat @ (_BOX_CORNERS * size).T).T + xpos
            rel = corners_w - cam_pos
            corners_c = (cam_rot.T @ rel.T).T
            z = -corners_c[:, 2]
            in_front = z > 1e-4
            if not np.any(in_front):
                continue
            u = cx + f * corners_c[in_front, 0] / z[in_front]
            v = cy - f * corners_c[in_front, 1] / z[in_front]
            pts = np.stack([u, v], axis=1)
            # keep points that could affect the image (allow a little margin)
            if np.all((pts[:, 0] < -2) | (pts[:, 0] > w + 1) | (pts[:, 1] < -2) | (pts[:, 1] > h + 1)):
                continue
            _fill_hull(mask64, np.round(pts).astype(np.int32))

        return _finalize_mask(mask64)

    def rgb(self, data) -> np.ndarray:
        """Head-camera RGB (64, 64, 3) uint8, for visualization only. Needs GL."""
        import mujoco
        if self.rgb_renderer is None:
            self.rgb_renderer = mujoco.Renderer(self.model, height=SEG_RENDER_SIZE, width=SEG_RENDER_SIZE)
        self.rgb_renderer.update_scene(data, camera=self.camera)
        return self.rgb_renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.rgb_renderer is not None:
            self.rgb_renderer.close()
            self.rgb_renderer = None


DEFAULT_YOLO_WEIGHTS = "yolo26n-sem-ade20k.pt"
YOLO26_SEG_FALLBACK = "yolo26n-seg.pt"
YOLO11_SEG_FALLBACK = "yolo11n-seg.pt"

_DETECT_ONLY_BASENAMES = {
    "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
}

# Surfaces the policy should treat as free (occupancy 0). Everything else is obstacle.
# ADE20K paints the whole table (top + legs) as "table"/"desk", unlike COCO instance-seg.
ADE20K_FREE_NAMES = {
    "floor", "rug", "ceiling", "wall", "sky", "windowpane",
    "painting", "mirror", "grass", "earth", "water", "mountain",
    "sea", "field", "sand", "river", "road", "sidewalk", "path",
    "runway", "hill", "lake", "land", "dirt track",
}
CITYSCAPES_FREE_NAMES = {"road", "sidewalk", "sky", "terrain"}


def refuse_detect_checkpoint(weights: str) -> None:
    """Raise if `weights` is a known detect-only file (no masks)."""
    import os

    path = os.path.normpath(str(weights)).replace("\\", "/")
    name = os.path.basename(path).lower()
    if name in _DETECT_ONLY_BASENAMES:
        raise RuntimeError(
            f"{weights} is a detect checkpoint (no masks). "
            f"Need semantic {DEFAULT_YOLO_WEIGHTS} or instance {YOLO26_SEG_FALLBACK} "
            f"(or {YOLO11_SEG_FALLBACK}). Not yolov5n.pt / best.pt."
        )
    if name == "best.pt" and "functions/model" in path.lower():
        raise RuntimeError(
            f"{weights} is onboard YOLOv11 detect (age/gender), not segmentation. "
            f"Pass {DEFAULT_YOLO_WEIGHTS} or {YOLO26_SEG_FALLBACK}."
        )


def class_name_map(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def free_names_for(names) -> set[str]:
    vals = set(class_name_map(names).values())
    if "table" in vals or "desk" in vals or "floor" in vals:
        return ADE20K_FREE_NAMES
    if "road" in vals and "sidewalk" in vals:
        return CITYSCAPES_FREE_NAMES
    return ADE20K_FREE_NAMES


def semantic_obstacle_mask(
    class_map: np.ndarray,
    names,
    free_names: set[str] | None = None,
) -> tuple[np.ndarray, list[tuple[str, float, tuple[int, int, int]]]]:
    """(H,W) int class ids -> uint8 obstacle mask {0,255} plus per-class labels."""
    name_of = class_name_map(names)
    if free_names is None:
        free_names = free_names_for(names)
    h, w = class_map.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    labels: list[tuple[str, float, tuple[int, int, int]]] = []
    n_pix = float(h * w) if h * w else 1.0
    for cid in np.unique(class_map):
        cid_i = int(cid)
        cname = name_of.get(cid_i, str(cid_i))
        if cname in free_names:
            continue
        hit = class_map == cid_i
        full[hit] = 255
        color = YoloSegOccupancy._PALETTE[cid_i % len(YoloSegOccupancy._PALETTE)]
        labels.append((cname, float(hit.sum()) / n_pix, color))
    labels.sort(key=lambda t: t[1], reverse=True)
    return full, labels


def read_semantic_class_map(result, hw: tuple[int, int] | None = None) -> np.ndarray | None:
    """Pull (H,W) int class ids from an ultralytics semantic Result, or None."""
    sm = getattr(result, "semantic_mask", None)
    if sm is None:
        return None
    data = sm.data if hasattr(sm, "data") else sm
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data)
    if arr.ndim == 3:
        # (C,H,W) logits/probs vs (1,H,W) ids
        if arr.shape[0] > 4 and arr.shape[0] != arr.shape[1]:
            arr = arr.argmax(axis=0)
        else:
            arr = arr[0]
    if arr.ndim != 2:
        return None
    if hw is not None and arr.shape != hw:
        import cv2

        arr = cv2.resize(arr.astype(np.int32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.int32)


class YoloSegOccupancy:
    """Deploy adapter: YOLO26 semantic (default) or instance-seg -> (1, 64, 64).

    Default yolo26n-sem-ade20k.pt labels indoor stuff (table including legs).
    Pass yolo26n-seg.pt / yolo11n-seg.pt for COCO instance masks.
    """

    COCO_OBSTACLE_NAMES = {
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "chair", "couch", "bed", "dining table", "toilet", "tv",
        "laptop", "keyboard", "mouse", "cell phone", "book",
        "suitcase", "backpack", "handbag", "bottle",
        "cup", "potted plant", "bench",
        "refrigerator", "oven", "microwave", "sink",
    }

    def __init__(self, weights: str = DEFAULT_YOLO_WEIGHTS, conf: float = 0.25, imgsz: int = 320):
        from ultralytics import YOLO

        refuse_detect_checkpoint(weights)
        self.weights = weights
        self.model = YOLO(weights)
        task = getattr(self.model, "task", None)
        stem = os.path.basename(str(weights)).lower()
        if task is None:
            if "-sem" in stem:
                task = "semantic"
            elif "-seg" in stem:
                task = "segment"
        if task not in ("segment", "semantic"):
            raise RuntimeError(
                f"{weights} loaded as task={task!r}, need 'semantic' or 'segment'. "
                f"Use {DEFAULT_YOLO_WEIGHTS} (semantic) or {YOLO26_SEG_FALLBACK} (instance), "
                "not detect / best.pt / yolov5n.pt."
            )
        self.task = task
        self.conf = conf
        self.imgsz = imgsz
        names = self.model.names
        self._names = class_name_map(names)
        self.free_names = free_names_for(names)
        if isinstance(names, dict):
            self.keep_cls = {i for i, n in names.items() if n in self.COCO_OBSTACLE_NAMES}
        else:
            self.keep_cls = {i for i, n in enumerate(names) if n in self.COCO_OBSTACLE_NAMES}

    _PALETTE = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
        (255, 128, 0), (0, 128, 255), (128, 255, 0), (0, 64, 255),
    ]

    def encode_bgr(self, bgr: np.ndarray) -> np.ndarray:
        return self.encode_views(bgr)["occupancy"]

    def encode_views(self, bgr: np.ndarray, keep_all: bool = False) -> dict:
        """One YOLO forward pass → policy occupancy plus full-res vis tensors.

        Returns dict with:
          occupancy: (1, 64, 64) float32 {0,1}  (unchanged policy contract)
          full_mask: (H, W) uint8 {0,255} union of kept instance masks
          overlay:   (H, W, 3) BGR tinted instances + labels
          labels:    list of (name, conf, color)
        """
        import cv2

        h, w = bgr.shape[:2]
        overlay = bgr.copy()
        full_mask = np.zeros((h, w), dtype=np.uint8)
        labels: list[tuple[str, float, tuple[int, int, int]]] = []

        pred_kw = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.task != "semantic":
            pred_kw["retina_masks"] = False
        results = self.model.predict(bgr, **pred_kw)
        if not results:
            return self._pack_views(overlay, full_mask, labels)

        r0 = results[0]
        names = r0.names if getattr(r0, "names", None) is not None else self.model.names
        class_map = read_semantic_class_map(r0, hw=(h, w))
        if class_map is not None or self.task == "semantic":
            if class_map is None:
                return self._pack_views(overlay, full_mask, labels)
            # Occupancy never treats floor/wall/sky as obstacle, even with --keep-all.
            full_mask, labels = semantic_obstacle_mask(
                class_map, names, free_names=self.free_names
            )
            name_of = class_name_map(names)
            tint_ids = np.unique(class_map) if keep_all else [
                int(c) for c in np.unique(class_map)
                if name_of.get(int(c), str(int(c))) not in self.free_names
            ]
            for cid in tint_ids:
                cid_i = int(cid)
                color = self._PALETTE[cid_i % len(self._PALETTE)]
                inst = class_map == cid_i
                tint = np.zeros_like(overlay)
                tint[inst] = color
                overlay[inst] = cv2.addWeighted(overlay, 0.45, tint, 0.55, 0)[inst]
                ys, xs = np.where(inst)
                if len(xs):
                    cname = name_of.get(cid_i, str(cid_i))
                    cv2.putText(
                        overlay,
                        cname,
                        (int(xs.min()), max(12, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            return self._pack_views(overlay, full_mask, labels)
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None else np.zeros(0, dtype=int)
        confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.zeros(0, dtype=float)
        xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else None
        masks_np = r0.masks.data.cpu().numpy() if r0.masks is not None else None

        n = 0 if masks_np is None else len(masks_np)
        if n == 0 and xyxy is None:
            return self._pack_views(overlay, full_mask, labels)

        count = n if n else (0 if xyxy is None else len(xyxy))
        for i in range(count):
            cid = int(cls_ids[i]) if i < len(cls_ids) else -1
            if not keep_all and cid not in self.keep_cls:
                continue
            if isinstance(names, dict):
                cname = str(names.get(cid, cid))
            else:
                cname = str(names[cid]) if 0 <= cid < len(names) else str(cid)
            color = self._PALETTE[cid % len(self._PALETTE)]
            conf = float(confs[i]) if i < len(confs) else 0.0
            labels.append((cname, conf, color))

            inst = None
            if masks_np is not None and i < len(masks_np):
                m = masks_np[i]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                inst = m > 0.5
            elif xyxy is not None and i < len(xyxy):
                x1, y1, x2, y2 = xyxy[i].astype(int)
                inst = np.zeros((h, w), dtype=bool)
                inst[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            if inst is None:
                continue

            full_mask[inst] = 255
            tint = np.zeros_like(overlay)
            tint[inst] = color
            overlay[inst] = cv2.addWeighted(overlay, 0.45, tint, 0.55, 0)[inst]
            ys, xs = np.where(inst)
            if len(xs):
                cv2.putText(
                    overlay,
                    f"{cname} {conf:.2f}",
                    (int(xs.min()), max(12, int(ys.min()) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return self._pack_views(overlay, full_mask, labels)

    def _pack_views(self, overlay, full_mask, labels) -> dict:
        occ_src = (full_mask.astype(np.float32) / 255.0)
        return {
            "occupancy": self._resize(occ_src),
            "full_mask": full_mask,
            "overlay": overlay,
            "labels": labels,
        }

    @staticmethod
    def _resize(mask: np.ndarray) -> np.ndarray:
        import cv2

        small = cv2.resize(mask, (OCCUPANCY_SIZE, OCCUPANCY_SIZE), interpolation=cv2.INTER_AREA)
        return (small > 0.3).astype(np.float32)[np.newaxis, ...]
