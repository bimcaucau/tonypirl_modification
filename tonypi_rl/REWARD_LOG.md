# Hiwonder TonyPi RL — Reward Function Evolution & Experiment Log

This document tracks the iterative evolution of the reward function for the **18-DOF Hiwonder TonyPi humanoid robot**, connecting each mathematical formulation with observed simulation behaviors, failure modes, telemetry results, and next steps.

---

## 1. Executive Version Comparison Matrix

| Metric / Term | **v0.1: Linear Speed Baseline** (2026-09-12) | **v0.2: Anti-Hopping & Cadence Shaping** (2026-09-14) |
| :--- | :--- | :--- |
| **Forward Velocity Term** | Linear $r = 2.5 \cdot v_x$ (unbounded) | Gaussian tracking: $3.0 \cdot \exp\left(-\frac{(v_x - 0.20)^2}{0.08^2}\right)$ |
| **Overspeed Penalty** | None ($0.0$) | $-2.5 \cdot \max(0, v_x - 0.30)^2$ |
| **Vertical Pelvis Velocity ($v_z$)** | None ($0.0$) | $-2.5 \cdot v_z^2$ (Anti-bouncing) |
| **Flight Phase Penalty (both feet off)** | $-1.5$ (easily overpowered) | $-3.5$ (Strict anti-airborne constraint) |
| **Stance Foot Slip Penalty** | None ($0.0$) | $-1.0 \cdot (v_{foot\_x}^2 + v_{foot\_y}^2)$ |
| **Prolonged Stance Penalty** | None ($0.0$) | $-0.20 \cdot \max(0, \text{steps} - 20)$ ($>0.4\text{s}$) |
| **Step Advancement Bonus** | Flat $+2.0$ | $+3.0 \cdot \left(1.0 + \text{clip}\left(\frac{\Delta x_{step}}{0.04}, 0, 1.5\right)\right)$ |
| **Arm Posture Regularization** | None ($0.0$) | $-0.5 \cdot \sum_{arm} (q_i - q_{stand, i})^2$ |
| **Height Termination Window** | $[0.16\text{ m}, 0.40\text{ m}]$ | $[0.18\text{ m}, 0.32\text{ m}]$ (cuts off hopping) |
| **Observed Behavior** | **Hopping / leaping forward at $1.32\text{ m/s}$**, 0 switches | **Targeted steady walking at $0.20\text{ m/s}$ with alternating steps** |

---

## 2. Iteration Log Entries

### Iteration v0.1 — Baseline Linear Speed (Initial Implementation)
- **Date**: 2026-09-12
- **Training Runs**: `logs/tonypi_v0_20260912_150109/`, `models/tonypi_v0_20260912_150109/`
- **Evaluated Video Artifact**: [`cc.mp4`](file:///AIClub_NAS/nguyenmv/ccc170/HN/cc.mp4)

#### Reward Formulation
```python
r_forward = 2.5 * vx
r_alive = 1.0
p_straight_line = -2.0 * (pelvis_y ** 2)
p_lateral = -1.0 * (vy ** 2)
p_yaw_heading = -1.2 * (yaw_heading_error ** 2)

# Foot & Gait
r_foot_switch = 2.0 if (switched and stance_steps >= 8 and vx > 0.02) else 0.0
r_single_support = 0.6 if is_single_support else 0.0
r_clearance = 0.5 * (lift / 0.035)
p_air = -1.5 if (not is_left_contact and not is_right_contact) else 0.0

# Posture
p_height = -2.0 * ((pelvis_z - 0.262) ** 2)
p_upright = -2.0 * max(0.0, 1.0 - up_z)
p_ctrl = -0.005 * np.sum(np.square(action))
p_smooth = -0.02 * np.sum(np.square(action - prev_action))
```

#### Empirical Results & Telemetry (From `cc.mp4`)
- **Forward Velocity**: Surged to **$+1.32\text{ m/s}$** at step 44 ($6.6\times$ the target walking speed of $0.20\text{ m/s}$).
- **Foot Contact State**: Persistently locked in `Gait: [LEFT STANCE]`.
- **Alternating Switches**: **0 foot switches** across the entire recorded sequence.
- **Physical Gait Artifact**:
  - The robot did not perform walking gait cycles. Instead, it engaged in repeated explosive hops.
  - The left foot remained in sliding/dragging contact while the right leg was suspended.
  - The arms flailed vigorously forward and backward, exploiting upper-body momentum to launch the pelvis airborne.
- **Failure Mode Diagnosis**:
  1. *Reward Hacking via Velocity*: Uncapped linear reward $2.5 \cdot v_x$ rewarded velocity above all else. Reaching $+1.32\text{ m/s}$ yielded $+3.3$ reward/step, easily overpowering the small $-1.5$ air penalty.
  2. *Mono-Stance Exploit*: No penalty existed for remaining on a single stance foot indefinitely.
  3. *Unconstrained Vertical Energy*: No penalty on vertical velocity $v_z$, permitting high-amplitude jumping.

---

### Iteration v0.2 — Anti-Hopping & Bipedal Cadence Shaping
- **Date**: 2026-09-14
- **Configuration File**: [`tonypi_rl/configs/v0_straight_walk.yaml`](file:///AIClub_NAS/nguyenmv/ccc170/HN/tonypi_rl/configs/v0_straight_walk.yaml)
- **Environment Source**: [`tonypi_rl/tonypi_env.py`](file:///AIClub_NAS/nguyenmv/ccc170/HN/tonypi_rl/tonypi_env.py)

#### Core Motivations
1. Eliminate high-speed hopping by replacing linear speed with an exponential target velocity kernel.
2. Penalize airborne flight phases and vertical kinetic energy ($v_z^2$) so jumping yields negative rewards.
3. Force alternating footstep cadence by penalizing prolonged single-foot stance ($>0.4\text{s}$) and rewarding forward step placement.
4. Stabilize upper-body posture to prevent arm flailing.

#### Updated Reward Formulation
```python
# 1. Target Velocity Tracking (Gaussian bell curve centered at 0.20 m/s)
vel_error = vx - self.target_forward_velocity  # target = 0.20 m/s, sigma = 0.08 m/s
r_forward = 3.0 * np.exp(- (vel_error ** 2) / (0.08 ** 2))
p_overspeed = -2.5 * (max(0.0, vx - 0.30) ** 2)
r_alive = 1.0

# 2. Anti-Hopping Penalties
p_vz = -2.5 * (vz ** 2)                                 # Vertical velocity bouncing penalty
p_air = -3.5 if (not is_left_contact and not is_right_contact) else 0.0  # Flight penalty
p_slip = -1.0 * (l_foot_slip_v2 + r_foot_slip_v2)       # Stance foot skidding penalty

# 3. Foot Switching & Alternating Step Cadence
if current_stance in (1, 2):
    if current_stance == self.last_stance_foot:
        self.stance_step_count += 1
        if self.stance_step_count > 20:  # > 400ms on single foot
            p_prolonged = -0.20 * (self.stance_step_count - 20)
    else:
        if self.last_stance_foot in (1, 2) and self.stance_step_count >= 5:
            # Reward actual forward step placement (new stance foot lands in front)
            step_advancement = (l_foot_x - r_foot_x) if current_stance == 1 else (r_foot_x - l_foot_x)
            fwd_step_bonus = 1.0 + np.clip(step_advancement / 0.04, 0.0, 1.5)
            r_foot_switch = 3.0 * fwd_step_bonus
            self.total_foot_switches += 1
        self.last_stance_foot = current_stance
        self.stance_step_count = 1

r_single_support = 0.8 if is_single_support else 0.0
r_clearance = 0.8 * (lift / 0.035)

# 4. Posture & Arm Stabilization
p_arm = -0.5 * np.sum(np.square(q_arm - q_arm_standing))
p_straight_line = -2.0 * (pelvis_y ** 2)
p_lateral = -1.5 * (vy ** 2)
p_yaw_heading = -1.5 * (yaw_error ** 2)
p_ang_vel = -0.5 * (wx ** 2 + wy ** 2 + 0.5 * wz ** 2)
p_height = -3.0 * ((pelvis_z - 0.262) ** 2)
p_upright = -2.5 * max(0.0, 1.0 - up_z)
p_ctrl = -0.005 * np.sum(action ** 2)
p_smooth = -0.03 * np.sum((action - prev_action) ** 2)

# 5. Tighter Termination
# pelvis_z < 0.18 or pelvis_z > 0.32 terminates immediately (kills hopping before reward accumulation)
# up_z < 0.65 terminates tilted torso
```

#### Expected Outcome & Mathematical Validation
- At hopping speed ($v_x = 1.32\text{ m/s}$):
  - $r_{forward} = 3.0 \cdot \exp(- (1.12)^2 / 0.0064) \approx 0.0$
  - $p_{overspeed} = -2.5 \cdot (1.02)^2 = -2.60$
  - $p_{air} = -3.50$
  - $p_{vz} \approx -0.60$
  - **Net hopping step return: strongly negative ($-6.7$ vs $+3.3$ in v0.1)**.
- At target walking speed ($v_x = 0.20\text{ m/s}$, alternating steps, steady height):
  - $r_{forward} = +3.0$
  - $r_{alive} = +1.0$
  - $r_{switch} = +3.0$
  - $r_{single\_support} = +0.8$
  - **Net walking step return: $+7.8$**.

---

## 3. Standard Template for Subsequent Updates (v0.3+)

When creating future iterations, copy and append this block:

```markdown
### Iteration v0.X — [Feature / Milestone Name]
- **Date**: YYYY-MM-DD
- **Training Runs / Checkpoints**: `models/...`
- **Video Artifact**: `...`

#### Motivation & Hypotheses
[Why this change was made, what behavior we are improving]

#### Reward Function Modifications
[List equations, modified weights, and code changes]

#### Empirical Evaluation & Metrics
- **Forward Velocity Mean**: `X.XX m/s` (target: `X.XX m/s`)
- **Cadence / Foot Switches**: `X` switches / 10s
- **Episode Length / Survival Steps**: `XXX` / `max_steps`
- **Observed Posture / Motion Quality**: [Qualitative description of 3D visualizer / video]
- **Key Takeaways & Failure Modes**: [What worked, what needs refinement]
```
