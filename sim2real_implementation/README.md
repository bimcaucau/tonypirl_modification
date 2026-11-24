# TonyPi nav sim-to-real

High-level nav deploy (`Discrete(7)` action-group clips). Walking / 18-DoF PPO is out of scope.

| File | What it is |
|---|---|
| [SIM2REAL_NAV.md](SIM2REAL_NAV.md) | Source of truth for the observation/action contract and module list |
| [HARDWARE.md](HARDWARE.md) | On-robot APIs, frames, IDs, units |
| [BRINGUP.md](BRINGUP.md) | Ordered live tests — do not skip phases |
| [AGENT_FEEDBACK.md](AGENT_FEEDBACK.md) | **Read this before changing code.** Bugs in the current tree, Pi gates, vis contract |
| [YOLO26_SEG.md](YOLO26_SEG.md) | **Next implement:** YOLO26n-seg occupancy. The 2026-09-18 debug PNG is HSV, not YOLO |
| [GROUND_LOOK.md](GROUND_LOOK.md) | Look down before a step (wires / low blocks) and correct distance. Laptop tests pass; robot checks are the remaining work |
| [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) | What already passed (Phase 0 laptop) |

Checkpoint (do not swap without updating the spec):

```
models_ckpt/new_expert_implementation/tonypi_final_model.zip
```

That zip is **high-level nav** (`Dict{vec:(18,), occupancy:(1,64,64)}` → `Discrete(7)` names). It is **not** the 18-DoF walking policy.

## Live 3-pane segmentation

Default occupancy net is **YOLO26n semantic ADE20K** (`yolo26n-sem-ade20k.pt`) so table legs / furniture stuff are occupied, not only COCO instance blobs. Instance-seg fallback: `--yolo-weights yolo26n-seg.pt`. See [YOLO26_SEG.md](YOLO26_SEG.md).

```
# laptop fake scene
python -m sim2real_implementation.view_segmentation --backend fake --mode color --max-frames 5 --save-dir /tmp/tonypi_seg

# robot occupancy vis (stop tonypi.service first; this script does not walk)
python3 -m sim2real_implementation.view_segmentation \
  --backend tonypi --mode yolo --http-port 8080
# instance-seg fallback: --yolo-weights yolo26n-seg.pt
# laptop browser: http://<ROBOT_IP>:8080

# HSV tuner only (painted-red boxes). Do not use this as occupancy in the office scene.
python3 -m sim2real_implementation.view_segmentation --backend tonypi --mode color --http-port 8080
```

Panes: camera | segmentation overlay | red=obstacle / green=road. Do not pass onboard `best.pt` or `third_party/yolov5/yolov5n.pt` (both detect-only).

Do not run the viewer at the same time as `deploy_nav` — the USB camera is exclusive.

## Walk to a relative goal

Robot is the origin when the script starts. `+x` forward, `+y` left (same as sim). Each step uses the live semantic occupancy mask.

```
python3 -m sim2real_implementation.go_to --dist 0.8 --angle-deg 0
python3 -m sim2real_implementation.go_to --dist 0.6 --angle-deg 30
python3 -m sim2real_implementation.go_to --xy 0.6 -0.2
# laptop browser, while it walks: http://<ROBOT_IP>:8080/
# disable the stream: --no-view
```
