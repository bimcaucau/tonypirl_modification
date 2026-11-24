# Plan: replace HSV / YOLOv5n with YOLO26n-seg occupancy

**Status (2026-09-19):** default occupancy is **semantic** `yolo26n-sem-ade20k.pt` (table/desk stuff includes legs). Instance-seg fallback `--yolo-weights yolo26n-seg.pt`. Floor/wall/sky stay occupancy 0. Do not rewrite walking. Do not call `expert_action()` on the Pi. Do not load onboard `Functions/model/best.pt` or `third_party/yolov5/yolov5n.pt`.

After each phase, append to `IMPLEMENTATION_LOG.md`.

---

## 0. What is actually broken (read the debug image)

File: `sim2real_implementation/debug/Screenshot 2026-09-18 145737.png`

HUD on that frame:

```
color 640x480  fps=36.9  enc=10.5ms  occ_mean=0.025  det=hsv  occ=64x64
```

That run is **`--mode color` (HSV)**, not YOLO. 10.5 ms encode is HSV; a nano-seg net on a Pi is tens–hundreds of ms. `occ_mean=0.025` is a handful of red pixels.

What the three panes show:

| Pane | What you see | Why |
|---|---|---|
| 1 camera | Desk, two people, hand, chairs, floor | Good camera stream |
| 2 overlay | Speckled orange on skin / red bracelet / a bit of clothing | Default HSV `[0,80,80]–[12,255,255]` matches **red-ish**, not “obstacle” |
| 3 binary | Green “road” everywhere, red crumbs on the hand | Same HSV mask downsampled. Table, people, chairs = free space |

The policy occupancy inset (64×64) is almost empty. A nav policy that treats occupancy=1 as obstacle will **walk into the desk**.

This is **not** “YOLOv5n is too weak”. YOLOv5n was never in this loop.

---

## 1. What lives in `tonypi_inside/` (do not reuse)

| Path | Task | Classes | Use for nav occupancy? |
|---|---|---|---|
| `TonyPi/Functions/model/best.pt` + `yolo11_detection.py` | **Detect** | age/gender (`female_child` … `male_elder`) | **NO.** Boxes only, wrong taxonomy |
| `third_party/yolov5/yolov5n.pt` (also s/m) | **Detect** | COCO 80 | **NO.** No instance masks. YAML `models/yolov5n.yaml` is detect. Seg yaml exists (`models/segment/yolov5n-seg.yaml`) but **no `yolov5n-seg.pt` is in the tree** |
| `TonyPi/Functions/ColorDetect.py` | HSV red/green/blue **ball** | 3 colors | **NO.** Same class of bug as current `--mode color` |
| `TonyPi/Functions/HolisticDetect.py` | MediaPipe body | skeleton | **NO.** Not a floor occupancy map |
| `TonyPi/Functions/ApriltagDetect.py` | Tag id + pixel | tags | Goal only, already in `goal.py` |

Stock apps import `ultralytics.YOLO` the same way we will. Copy that **style** (`YOLO(path); results = model.predict(...)`), not those **weights**.

Official nano-seg weights for this plan:

```
yolo26n-seg.pt
```

Docs: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [segment task](https://docs.ultralytics.com/tasks/segment/). Ultralytics name is **YOLO26**, filename `yolo26n-seg.pt` (not `yolov26n`). If a user says “yolov26n”, they mean this file.

Load:

```python
from ultralytics import YOLO
model = YOLO("yolo26n-seg.pt")  # downloads on first use if missing
results = model.predict(bgr, conf=0.25, imgsz=320, verbose=False)
# results[0].masks.data  (N,H,W)
# results[0].boxes.cls
```

API is the same as YOLO11-seg. `YoloSegOccupancy` in `tonypi_rl/perception.py` already speaks that API; it currently defaults to `yolo11n-seg.pt`.

---

## 2. Locked contracts (do not change)

- Policy obs: `Dict{ occupancy: (1,64,64) float32 in {0,1}, vec: (18,) }`
- Occupancy **1 = obstacle, 0 = free/road**. Pane 3 of the viewer: red=obstacle, green=road.
- `OccupancyEncoder.encode()` still returns **only** `(1,64,64)`. Overlays stay in `encode_views()`.
- Action names frozen. No 18-DoF walk. DummyNavEnv loader only — do not `import tonypi_rl.tonypi_nav_env` on the Pi.
- Gyro zeros. ToF `.so` load stays `hardware.import_vl53l0x()`.
- Do not vendor `third_party/yolov5` into the deploy path.

Sim occupancy (training) is a **pinhole fill of mocap boxes**, not COCO instances. Real YOLO masks (person/chair/table) are a domain gap. That is acceptable for v1: the CNN sees “stuff in the image”, which is still closer to sim than HSV skin speckles. Do **not** start a custom dataset until Phase 5 (optional).

---

## 3. Target architecture

```
head camera BGR 640x480
        │
        ▼
 YOLO26n-seg  (ultralytics, CPU, imgsz=320, conf=0.25)
        │
        ├─ instance masks (kept classes) ── union ── full-res uint8 mask
        │                                              │
        │                                              ├─ overlay (pane 2)
        │                                              ├─ red/green binary (pane 3)
        │                                              └─ INTER_AREA → (1,64,64) occupancy
        └─ refuse if results[0].masks is always None (detect-only checkpoint)
```

Default deploy occupancy mode becomes **`yolo`**, weights **`yolo26n-seg.pt`**. Keep `--mode color` as a fallback after HSV is tuned; it is not the occupancy source of truth.

---

## 4. Files to change

### 4.1 `tonypi_rl/perception.py` — `YoloSegOccupancy`

- Default `weights="yolo26n-seg.pt"`.
- After `YOLO(weights)`, **assert a seg model**:
  - If `self.model.task != "segment"` (or first `predict` on a dummy frame has `masks is None` and boxes exist), raise `RuntimeError` naming the file and saying “need *-seg.pt, not detect / best.pt / yolov5n.pt”.
- Refuse paths ending in `best.pt` or equal to `yolov5n.pt` / `yolov5s.pt` / `yolov5m.pt`.
- Expand `COCO_OBSTACLE_NAMES` for this lab (the debug scene is people + tables + chairs):

```
person, chair, couch, bed, dining table, bench,
laptop, tv, keyboard, mouse, cell phone, book,
backpack, handbag, suitcase, bottle, cup, potted plant,
refrigerator, oven, microwave, sink, toilet,
bicycle, car, motorcycle, bus, truck
```

Keep `--keep-all` / `keep_all=True` for the viewer so you can see ignored classes. Deploy default stays the obstacle set (not `toothbrush`).

- Predict kwargs: `conf`, `imgsz`, `verbose=False`. Add `retina_masks=False` (faster; we already resize masks to camera HxW). Do not pass `device=0` on the Pi.
- One forward pass in `encode_views`; `encode_bgr` keeps calling it. Do not double-predict.

### 4.2 `sim2real_implementation/occupancy_real.py`

- Default `yolo_weights="yolo26n-seg.pt"`.
- When `mode=="yolo"`, if weights look like detect-only, fail at init (do not silently fall back to HSV).
- Pass `keep_all` through as today.

### 4.3 `sim2real_implementation/view_segmentation.py`

- Default `--mode yolo` (today: `color` — that is why the screenshot is HSV).
- Default `--yolo-weights yolo26n-seg.pt`.
- HUD must print **weights basename + task**, e.g. `yolo26n-seg imgsz=320`, not just `det=hsv`.
- Keep `--mode color` for the HSV tuner path.
- Still `TonyPiHardware(dry_run=True, allow_missing_tof=True)` — vis does not walk.
- Refuse `best.pt` (already does). Also refuse `yolov5n.pt`.

### 4.4 `sim2real_implementation/configs/deploy.yaml`

```yaml
occupancy: yolo
yolo_weights: yolo26n-seg.pt
yolo_imgsz: 320
yolo_conf: 0.25
yolo_keep_all: false
```

Leave HSV keys in the yaml for fallback; they are unused when `occupancy: yolo`.

### 4.5 `sim2real_implementation/requirements-deploy.txt`

```
ultralytics>=8.4.0
```

Pin a version that actually ships YOLO26 (released Jan 2026). On the Pi:

```bash
python3 -c "from ultralytics import YOLO; m=YOLO('yolo26n-seg.pt'); print(m.task, m.names[0])"
```

Must print `segment` and `person`. If it downloads `yolo11n-seg` or fails to resolve `yolo26n-seg.pt`, upgrade ultralytics; do not silently swap weights.

### 4.6 Tests (`tests/test_obs_contract.py`)

- Keep existing HSV / encode_views tests.
- Add `test_refuse_detect_checkpoint`: constructing `YoloSegOccupancy(weights=...yolov5n.pt)` or a fake detect yaml **raises** (mock `YOLO` if the real file is huge).
- Do **not** require a GPU or a live download in CI. If `yolo26n-seg.pt` is on disk, an optional test may run one predict on the debug screenshot and assert `occ_mean > 0.05` and that pane-3 red covers the foreground person (not only the hand). Mark it `@unittest.skipUnless(os.path.exists(weights))`.

### 4.7 Docs

- `README.md`: default vis command uses `--mode yolo --yolo-weights yolo26n-seg.pt`.
- `AGENT_FEEDBACK.md`: note that the 2026-09-18 debug PNG is HSV, not YOLO.
- `IMPLEMENTATION_LOG.md`: Phase “YOLO26n-seg occupancy” with the gate commands below.

Do **not** copy `third_party/yolov5` into `sim2real_implementation/`.

---

## 5. Raspberry Pi bring-up (required)

Layout on this robot (siblings under `$HOME`):

```
~/TonyPi/
~/sim2real_implementation/
~/tonypi_rl/          # required: YoloSegOccupancy lives here
~/yolo26n-seg.pt      # or let ultralytics download into ~/.ultralytics
```

```bash
export PYTHONPATH="$HOME:$PYTHONPATH"
export TONYPI_HOME="$HOME/TonyPi"
sudo systemctl stop tonypi
cd "$HOME"

# 1. ultralytics that knows YOLO26
pip3 install -U "ultralytics>=8.4.0"

# 2. sanity: must be SEGMENT
python3 -c "from ultralytics import YOLO; m=YOLO('yolo26n-seg.pt'); print(m.task)"

# 3. viewer — same camera as the bad screenshot
python3 -m sim2real_implementation.view_segmentation \
  --backend tonypi --mode yolo --yolo-weights yolo26n-seg.pt \
  --yolo-imgsz 320 --http-port 8080 --save-dir ~/seg_debug
```

Open `http://<ROBOT_IP>:8080`. Capture a new PNG into `sim2real_implementation/debug/`.

**Pass (same scene as the old screenshot):**

- HUD contains `yolo26n-seg` (not `det=hsv`).
- Pane 2: colored instance blobs on **people** (and ideally chairs / table), with class labels.
- Pane 3: large red regions on people/furniture; floor/walls mostly green. `occ_mean` clearly above the old `0.025` (expect ≳ 0.15 in that crowded lab).
- Encode latency logged. Budget: **< 250 ms/frame** on Pi 4/5 at `imgsz=320`. If > 400 ms, go to §7 (export) before changing the policy.

**Fail:**

- HUD still `det=hsv` → you did not switch default mode / did not pass `--mode yolo`.
- `masks is None` → you loaded a detect checkpoint. Stop.
- Only the hand is red → still HSV, or conf too high and nothing else detected.
- Whole frame red → `keep_all` plus background classes, or mask resize bug. Fix before deploy.

Do not run `deploy_nav` until the viewer PASSes.

---

## 6. Wire into deploy (only after viewer PASS)

```bash
python3 -m sim2real_implementation.deploy_nav \
  --backend tonypi \
  --occupancy yolo \
  --yolo-weights yolo26n-seg.pt \
  --yolo-imgsz 320 \
  --goal 0.4 0.0 \
  --max-steps 8 \
  --arrive 0.25
```

Still leash. `hw.start()` must still fail-closed on camera. YOLO latency sits **inside** the step before `play()`; do not add a 0.4 s sleep.

If `occ_mean` in `nav_log.jsonl` is ~0 on a cluttered scene, the policy will `go_forward` into furniture. Treat that as a P0 — do not “tune the policy”.

---

## 7. If Pi is too slow (optional, same masks)

Do this only if viewer PASS on quality but latency > 400 ms.

1. On the laptop (x86 is fine for export; run the converted model **on the Pi**):

```python
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="ncnn", imgsz=320)
# or format="onnx"
```

2. Load the exported dir/file with the same `YOLO(...)` constructor (`YOLO("yolo26n-seg_ncnn_model")`).
3. Keep `encode_views` reading `result.masks`. If NCNN drops masks, **do not ship it**; stay on `.pt` CPU or try ONNX.
4. Do not export TensorRT unless the board has a GPU you have already used for other Hiwonder demos.

`imgsz` ladder if still slow: 320 → 256. Do not go below 256; 64×64 occupancy from a 160 px net is mush.

---

## 8. Class filter vs this lab

The debug frame is **people + table + chairs**, not a red cardboard box. COCO `person`, `chair`, `dining table` are the right labels.

- Viewer: start with `--keep-all` once to list every name YOLO assigns. Write the names into the log.
- Deploy: obstacle allow-list from §4.1. If the table is labeled `dining table` and masked, good. If it is missed, add that name; do not lower conf below 0.15 without looking at false-positive floor.
- Cardboard boxes (sim obstacles) are often `suitcase` / missed. If the **real** eval is boxes on the floor, collect 20 frames and check names. Expanding `keep_cls` is cheaper than training.

HSV stays available (`--mode color` + `sensor_probe --tune-hsv`) for a painted-red obstacle course. It is not the default.

---

## 9. Optional Phase 5 — custom obstacle/road (only if COCO fails)

Only if viewer PASS still leaves the table/floor wrong after allow-list tweaks.

- Label 50–100 head-camera frames: binary obstacle vs free (or instance “obstacle”).
- `yolo segment train model=yolo26n-seg.pt data=our.yaml epochs=50 imgsz=320`.
- Ship `runs/segment/weights/best.pt` **as a seg checkpoint**, never overwrite `Functions/model/best.pt`.
- This is a new weights file, e.g. `tonypi_occ_yolo26n-seg.pt`. Update yaml. Do not change obs shape.

---

## 10. Suggested implement order

1. Refuse detect checkpoints in `YoloSegOccupancy.__init__`.
2. Default weights `yolo26n-seg.pt`; expand `COCO_OBSTACLE_NAMES`; HUD prints model id.
3. Default viewer + `deploy.yaml` occupancy `yolo`.
4. Laptop: `view_segmentation --backend fake` still works; `--backend webcam --mode yolo` on a person in view → red body in pane 3.
5. Pi: §5 viewer gate, save a new debug PNG next to the old one.
6. Pi: §6 leash `deploy_nav`.
7. Only then §7 export or §9 fine-tune.

---

## 11. Do not

- Do not `git submodule` or import `tonypi_inside/third_party/yolov5`.
- Do not use `yolov5n.pt`, `yolov5n-seg` from a random URL, or `Functions/model/best.pt`.
- Do not train YOLO inside PPO / DummyNavEnv.
- Do not change `(1,64,64)` or Discrete(7).
- Do not enable `balance_ankle`.
- Do not run the viewer and `deploy_nav` at the same time (exclusive camera).
- Do not “fix” occupancy by tightening HSV on this office scene.

---

## 12. Success picture (compare to the current PNG)

Old: green pane 3, red crumbs on a hand, `det=hsv`, `occ_mean=0.025`.

New: people (and desk/chairs if detected) are **filled red** in pane 3, floor stays green, HUD `yolo26n-seg`, `occ_mean` in the tenths, encode time within budget. That occupancy is what `deploy_nav --occupancy yolo` must feed the zip.
