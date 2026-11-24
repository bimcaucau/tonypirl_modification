# Ground look — wires, low blocks, and distance

**Status:** implemented and unit-tested on the laptop (`tests/test_ground_look.py`, 17 tests). **Not yet run on the robot.** This file is the handoff. Do not reopen the design unless a live check below fails.

The policy is unchanged. Occupancy is still `(1, 64, 64)` from the **level** head. Walking is still the `.d6a` clips. Nothing here calls `expert_action()`.

---

## Why the robot trips, and why its distances are wrong

Two sensors never see a wire or a small block on the floor:

- The head is held at pulse 1500 so the camera matches the MuJoCo `head_camera` (level, +X). With that pitch the **center ray is parallel to the floor** and never hits it. The bottom of the image meets the floor about **half a metre** out. Anything closer is under the frame.
- The VL53L0X is one chest-height beam (`tof_0` at z ≈ 0.045 m in the pelvis, aimed +X, cutoff 1.2 m). It flies over the same objects and reports clear. `vec[3]` is that number. The policy then picks `go_forward`.

Goal distance is a separate error. `goal.py` adds a fixed primitive per clip (`go_forward` = 0.08 m). The real step is not 0.08 m, so `dist` drifts and the robot stops early or walks past the goal.

---

## What shipped

```
level frame  →  policy (unchanged obs)  →  safety
                      │
                      ▼
        head down (pulse 1100, ~36°)  center, then left, then right
                      │
                      ├─ texture clutter (black-hat wires + blobs off the floor color)
                      ├─ semantic mask, but only if it does NOT fill the footprint
                      │     (a filled mask is a mislabeled floor, not a wire)
                      ▼
        project pixels onto z = floor with the pinhole + known pitch
                      │
                      ├─ footprint occupied → replace the clip (avoid_action)
                      ├─ nearest clutter in the forward corridor → min(ToF, range)
                      │     and re-query the policy once with that shorter range
                      └─ phase-correlate this look-down frame with the previous
                            one → replace the last translation primitive
                      │
                      ▼
        head back to 1500, drop the live-view frame grabbed while looking down
                      │
                      ▼
        play the (possibly replaced) clip
```

| Piece | Where |
|---|---|
| Geometry, clutter, veto, floor-shift | `ground_look.py` |
| Called from the walk loop | `deploy_nav.py` (after `safety.filter_action`, before `hw.play`) |
| `set_head(pitch, yaw, ms)` | `hardware.py` — PWM 1 pitch, PWM 2 yaw. Below 1500 is down (`hand_nod.py`) |
| Measured step replaces the primitive | `GoalSource.correct_last_translation` (dead-reckon only, translations only) |
| Live 3-pane must not become the next policy frame | `LiveTriple.invalidate` |
| Flags | `--no-ground-look`, `--no-look-sweep`, `--look-pulse`, `--look-yaw-sign`. `go_to.py` forwards the first two |
| Defaults | `configs/deploy.yaml`: `ground_look: true`, `look_pulse: 1100` (~36° down; 900 framed the shins and every step became `back_fast`), `look_sweep: true`. Side and head-turn poses are aimed at the floor ahead, not at the feet. `look_pitch_sign: -1`, `look_yaw_sign: 1` |

A wire across the front becomes `turn_left` / `turn_right` toward the side with less clutter, else `back_fast`, else `stand`. The reason in the jsonl is `ground_look`. The policy action is still logged as `raw_action`.

Footprints (pelvis frame, metres) live in `FOOTPRINTS` at the top of `ground_look.py`. Forward step is x ∈ [0.14, 0.36], |y| ≤ 0.09. Nearer than 0.14 is the robot's own feet and is ignored on purpose.

Floor-shift is accepted only when phase-correlation response ≥ 0.15, the action between the two looks was a translation, and the result is at most 0.20 m forward / 0.15 m sideways. A blank floor returns nothing and the primitive stands. Turns are not corrected.

Script mode (`--script`) does **not** look. It is a clip calibration tool.

---

## Do not touch

- The policy zip, `vec` layout, occupancy shape, action-name order.
- `tonypi_rl/tonypi_nav_env.py`, MuJoCo XML, `.d6a` files.
- Head pose **during the policy frame**. It must be pulse 1500 / 1500. The look is only between the decision and the clip.
- Feeding the look-down mask into `obs["occupancy"]`. That is a different camera than training, and it will not "just work."

---

## Live checks, in order

Stop `tonypi.service` first. Same as `BRINGUP.md`.

### 1. The head actually looks at the floor

```bash
python3 -m sim2real_implementation.deploy_nav \
  --backend tonypi --dry-run --occupancy yolo \
  --goal 0.8 0 --max-steps 1 --http-port 8080 \
  --allow-missing-tof
```

Watch the 3-pane stream. You should see floor ahead of the robot, never the chest, shins, or feet. If the body fills the frame, the veto calls it a wire and the only clip left is `back_fast`. The side looks are the same shallow pitch, turned left and right, not a stare at the shoes. No clip plays (`--dry-run`). A wire on the floor ahead must log `ground_look`; a clear floor must not.

| What you see | Change |
|---|---|
| Pulse 1100 looks **up**, or into the chest | Servo 1 is reversed. Set `look_pitch_sign: 1` and `look_pulse: 1900` (the other side of 1500 by the same 400 µs). Do not only change the pulse: the ray math uses the sign. |
| Floor is in frame but the toes or shins fill it | Raise `look_pulse` toward 1200 (less tilt). Do not go back to 900. |
| Floor is only a strip at the bottom | Lower `look_pulse` toward 800. Stay ≥ 600. |
| Left yaw shows the right side of the room | Set `look_yaw_sign: -1`. Until this is right, a blocked step can turn **toward** the wire. |

### 2. Veto on a real wire, no veto on clear floor

Clear floor, one `--max-steps 3` dry-run. jsonl `ground_blocked` must be empty. If `go_forward` is blocked on a floor you can see is empty, the feet or a shadow are in the mask: raise x0, or raise `CLUTTER_FRACTION` / the black-hat threshold `18` in `clutter_mask`. Do not disable the look to "make it walk."

Then lay a cable across the next step. The same dry-run must log `safety_reason: ground_look` and an action other than `go_forward`, and `ground_range_m` should be in the 0.15–0.40 m band, not 1.2. `tof_fused` in the jsonl is what the re-query used; the raw `tof_m` stays the beam.

### 3. Step length

Tape 0.5 m on the floor. Run `go_to --dist 0.5` **without** `--no-ground-look`, on a floor with some texture (tile grout is enough; a featureless sheet is not). After a few steps the log lines `Floor shift corrected last step to dx=...`. Compare `odom_dx` to a tape measure for one `go_forward`.

| Result | Change |
|---|---|
| `odom_dx` within ~2 cm | Copy the mean into `configs/primitives.yaml` for `go_forward`. The corrector keeps fixing residual slip; the primitive then starts closer. |
| `odom_response` always under 0.15 | Floor is too blank. Leave the primitive. Do not lower the gate below 0.1; it will integrate noise. |
| Sign is wrong (it subtracts when the robot went forward) | Stop and fix `shift_to_motion` in `ground_look.py`. The unit test `test_downward_image_shift_is_a_forward_step` is the spec: texture moving **down** in the look-down image is +X. |

### 4. Only then, a real walk

`go_to --dist 0.8` with the cable in the way. Expect a turn or a back-up, not a trip. Put the cable beside the body, not only straight ahead: `turn_left` or `left_move_fast` must come back blocked in `ground_blocked`, and `go_forward` must not if the front is clear. `--no-look-sweep` if the five-pose look is too slow (forward only; the head turns are the slow part, about a second each, and without them a wire beside the robot is invisible again).

---

## If a later change is actually needed

- **Camera height.** Rays use pelvis height 0.267 m and camera `(0.08, 0, 0.110)` from `tonypi.xml`. If ranges in check 2 are consistently 30% high or low, scale `PELVIS_HEIGHT_M` in `ground_look.py` — do not fit a new network.
- **Intrinsics.** `GoalSource` loads `calibration_param.npz` and the look uses it, scaled from 640×480. A camera opened at another resolution needs that scale rechecked (`intrinsics_for`).
- **Thin wires the black-hat misses** (same color as the floor, thicker than 9 px). Lower the kernel only after check 2 shows a miss; a smaller kernel flags grout.
- **Do not** put a second policy on the look-down image. The veto is the whole point: the trained net never saw its own feet.
