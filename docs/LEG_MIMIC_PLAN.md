# Plan: walking and leg mimicry from the camera

Status: **PLAN ONLY, nothing implemented** (written 2026-10-24, at Seth's
request: "DO NOT WRITE ANY CODE for it yet"). This is the action plan Claude
follows when Seth says to start. Each phase ends at a gate. Don't start a
phase until the one before has passed its gate and Seth has said go.

Related:
- `docs/ARM_PRECISION_PLAN.md`: arms first.
- Robot runtime: `~/hrg_urdf/deploy/docs/ARCHITECTURE.md`.
- Policies: `~/hrg_urdf/isaac/eval/pinned/POLICIES.md`.

---

## 0. Why the legs are different from the arms

| | arms / head | legs |
|---|---|---|
| What moves them | hobby servos go to the angle sent | a **trained balance policy** (ONNX, 50 Hz) picks the joint targets from IMU + joint feedback |
| If a command is slightly wrong | the arm points slightly wrong | the robot can fall |
| Can we copy human angles directly? | yes (that is what teleop does now) | **no**: the angles would override balance; human and robot bodies, mass and feet differ; the camera can't see ground contact or weight shift |
| Hardware limits | servo range and slew | **no ankle roll** (legs are hip yaw/roll/pitch, knee, ankle pitch), so it can't balance on one foot for long; torque cap ~9.4 N·m in sim; small feet |

**Rule for every phase: the policy stays in charge of balance. The human
only ever sends *intent* (where to go, how low, how to lean, which reference
to follow), never raw leg joint angles.**

## 1. What exists already (reuse it, don't rebuild it)

- `run_policy.py`:
  - **Skills:** stand, walk, side, dance, loaded at once, picked from the
    command (`drive_step`).
  - **Commands:** vx/vy/wz go through `gamepad_command`, which handles the
    dead-man, accel/decel ramps and stick-to-cadence.
  - **Trained ranges:** walkers vx 0.15–0.25 m/s, side-step vy (0.15–0.30, or
    0.05–0.10 for `shuffle_lift`), wz ±0.3 rad/s **only while vx ≥ 0.15**
    (stand and side trained with wz 0).
  - **Safety:** dead-man LB, B stop, Y quit + land, sensor-age and overrun
    aborts, zero lock, target clamps.
- `hrg-teleop` (laptop): MediaPipe Pose at 30 fps. It gives 3D world
  landmarks centred on the hips, plus image-space landmarks for position and
  scale. UDP link with acks/RTT, about 13 ms round trip over Tailscale.
- Isaac Lab task `Isaac-Velocity-Flat-HRG-v0`
  (`~/IsaacLab/.../velocity/config/hrg/`).
  - `hrg_dance.py` plus the `HRG_DANCE*` terms in `flat_env_cfg.py` already
    do **reference-tracking rewards with a phase clock** (the groove emote).
    That's the starting point for imitation in Phase C.
  - Also: domain randomisation, latency buffer, reference state
    initialisation (`reset_to_gait_reference`).
- MuJoCo gates:
  - `isaac/mj_sim2sim.py`, `stand_eval.py`
  - `mj_drive.py` (keyboard driving of skills)
  - `arm_gesture_eval.py` (arms moving on a standing policy)

---

## Phase A: steer the existing walkers with your body (no training)

**Goal:** step forward and it walks forward; step sideways and it
side-steps; turn your shoulders and it turns; stand still and it stands.
Arms and head keep mimicking. Effort: **about 3–5 days.**

### A1. Laptop: body → intent (`hrg-teleop`)
1. **Home calibration (key `K`):** stand where you want "zero" and press K.
   Record:
   - hip-centre image x
   - torso size in pixels (shoulder-to-hip distance; a depth proxy)
   - shoulder-line yaw, from world landmarks relative to the camera
2. **Each frame:**
   - **lateral offset:** dx from image x, scaled to metres by torso size
   - **forward/back offset:** dz from torso-size ratio. Bigger = closer to
     the camera = forward when facing the robot. Mirror rules as for arms.
   - **heading:** dψ from shoulder yaw
3. **Zones, not proportional copying (v1):**
   - Inside a deadband (about ±0.15 m, ±12°): stand.
   - Outside it: send the trained command for that direction:
     - forward → vx ramped 0.15→0.25 by distance
     - sideways → vy in the side skill's range
     - turn → wz ±0.3, only together with forward, matching training
4. **Alternative (v2), "march in place":** detect alternating knee lifts
   (hip–knee angle) and their cadence, then map them to walk plus
   `--walk-hz-range` within the trained band. Try only after v1 works.
5. **Packet:** add to the existing UDP packet
   `"cmd": {"vx", "vy", "wz", "mode": "stand|walk|side"}` and keep
   `"enable"`. Arms continue as now.
6. **UI:**
   - a top-down "zone" diagram with your dot and the command arrow
   - big STAND / WALK / SIDE / TURN status
   - legs-on/off toggle (key `L`); legs off by default

### A2. Robot: network command input in `run_policy.py`
1. New option `--net-cmd PORT`.
   - A non-blocking UDP read once per tick; drain and keep the newest.
   - **Never block the 50 Hz loop.**
   - Measure the loop time before and after (`loop_ms` in telemetry).
2. **It is an alternative stick, not a new controller.**
   - It feeds `want` in `gamepad_command`, so the existing ramps, skill
     choice, turn-only-while-walking and heading hold all still apply.
3. **Safety, all of these:**
   - the gamepad dead-man LB must still be held: the operator stays in the
     loop
   - packet older than 0.3 s, or `enable=false` → command 0 → stand
   - gamepad B / Y and Ctrl+C unchanged, and they win over the network
   - command clipped to the trained ranges; walk only from a settled stand
     (like the dance entry rule)
   - the network can't arm the robot, change gains or start a run. Those
     stay on the Pi (`go` prompt).
4. **Arms:** `arm_teleop.py` keeps owning the servos. `try_walk.sh` gets a
   `TELEOP=1` mode that starts `arm_teleop.py` instead of `arm_pad.py`, plus
   `--net-cmd`. Then one laptop app drives both, while the walk arm swing is
   replaced by the human's arms.
5. Telemetry: record `cmd_src` (pad/net), packet age and seq gaps.

### A3. Test ladder (gate A)
1. Unit: zone logic on recorded MediaPipe sessions, replayed with no robot.
2. **MuJoCo:** add the same `--net-cmd` input to `mj_drive.py` and drive the
   sim robot from the camera for 10 minutes. No falls, stops within 1 s of
   stepping back in.
3. **Arms moving while walking, in sim:** the policies were trained with
   **rigid arms**.
   - Run `arm_gesture_eval`-style checks with teleop arm motion during walk
     and stand. Measure extra tilt and falls.
   - If it's bad, limit arm speed or amplitude while walking, or go to C0.
4. Hardware, robot hanging: commands appear, stops on every safety path
   (pull the Wi-Fi, close the laptop, release LB, press B).
5. Hardware on the floor with a spotter, 2 minutes, then 10 minutes.
   - **Gate A:** 10 minutes with no falls and no safety-path failures;
     latency from step to motion < 400 ms.

---

## Phase B: posture mimic while standing (training required)

**Goal:** crouch and it crouches; lean forward/sideways and it leans; feet
stay put. Effort: **about 1–2 weeks**, mostly training and sim2real.

1. **New stand-skill commands:** base-height offset (for example −6…0 cm) and
   torso pitch/roll targets (±10°).
   - The obs grows by 3. That's a new policy family, so `run_policy` needs
     the obs layout for it, the way 59/61/62 are handled now.
2. **Isaac (`flat_env_cfg.py`, new `HRG_POSTURE=1` mode):**
   - sample these commands; reward tracking of base height and torso
     attitude
   - keep the existing stand rewards: feet still, smoothness, torque
   - curriculum from 0 to full range
   - **randomise arm joint positions and velocities during training**, so
     the policy expects teleop arms (see C0)
3. **Laptop:** hip height from torso scale relative to the home calibration
   gives the height command; torso lean from world landmarks gives
   pitch/roll. Slow filter (0.5–1 Hz) and rate limits.
4. **Gate B:**
   - MuJoCo (`stand_eval` style): tracking error < 1.5 cm / 3°, no falls
     under pushes at the latency we've measured.
   - Hardware: hanging, then a spotter. Calm (stand median about 0.1°/step,
     like `s5_4000`).

---

## Phase C: whole-body imitation (research-level)

**Goal:** the robot follows your pose: steps where you step, knee lifts,
squats, turns. It looks like a mirror within what the hardware can do.
Effort: **weeks**. Follow the H2O / OmniH2O / HumanPlus approach, scaled to
this robot.

### C0. Prerequisite: policies that tolerate moving arms
- Retrain stand and walk with **randomised arm motion** (arm joints driven by
  random smooth trajectories and recorded teleop sessions), because
  today's policies assume rigid arms.
- Gate: same hardware quality as the current pinned policies, with arms
  moving.

### C1. Motion data and retargeting
1. **Sources:**
   - AMASS (large mocap set; check the licence: research use)
   - **our own recordings:** `hrg-teleop` gets a `--record` mode that saves
     MediaPipe world landmarks at 30 fps
2. **Retarget to HRG:**
   - scale by leg length
   - map hip/knee/ankle-pitch angles
   - hip yaw/roll from thigh direction
   - arms with the existing `retarget.py`
   - root trajectory scaled by leg-length ratio
3. **Filter for feasibility:**
   - no ankle roll → drop clips needing sustained single-leg balance or big
     lateral foot moves
   - joint limits, and torque within ~9.4 N·m in a quick MuJoCo replay
   - foot sliding
   - Keep a curated set of about 200–2000 clips.

### C2. Teacher policy (privileged, in Isaac)
- **Obs:** the robot's usual obs, plus the reference over the next few
  frames (joint targets, root velocity, key-point positions), plus
  privileged state (contacts, true base velocity).
- **Reward:**
  - joint/keypoint tracking, built on the `hrg_dance` reference-tracking
    terms
  - root velocity tracking
  - alive
  - smoothness, torque and foot-slip penalties, as in the current walkers
- **Domain randomisation** like the robust walkers:
  - latency 40–80 ms
  - friction, mass, gains ×0.30 sim2real setting
  - pushes
- **Reference state initialisation** from clip frames
  (`reset_to_gait_reference` pattern).

### C3. Student policy (deployable)
- Same obs as the hardware has, plus a **sparse** reference the camera can
  supply in real time: root velocity/heading, head and hand positions, and
  optionally knee/foot targets.
- Distil from the teacher with DAgger.
- Export to ONNX, then `run_policy` adds a `mimic` skill fed by the network
  reference (Phase A2 transport, a bigger packet).

### C4. Live pipeline
- Laptop: MediaPipe → retarget → reference buffer → UDP.
- Pi: `run_policy` mimic skill. The same safety as A2, plus automatic
  fallback to the stand skill when tracking confidence drops.
- **Latency matters more here:**
  - budget < 150 ms camera→command
  - consider a phone or 60 fps camera
  - the policy should be trained with that delay

### C5. Gate C
- MuJoCo: track 20 held-out clips, mean keypoint error < 6 cm, falls < 5%.
- Hardware: hanging, then spotter, then demo. Every time: emergency stop
  tested, tether or spotter.

---

## Ordering and decisions for Seth

| step | needs | effort | decision |
|---|---|---|---|
| Arm precision fixes | `docs/ARM_PRECISION_PLAN.md` | 1–2 days | — |
| A: body-steered walking | no training | 3–5 days | operator must hold LB? (recommended yes) |
| C0: arm-tolerant policies | Isaac training | ~1 week | needed before showing arms + walking together for long |
| B: posture mimic | training | 1–2 weeks | height/lean ranges |
| C: whole-body imitation | data, teacher, student | 4–8 weeks | AMASS licence, compute time |

**For the SHPE conference:** arms and head mimic while the robot stands on
the pinned stand policy. Phase A only if gate A passed well before the
event. Phases B/C are after the conference.

## Risks

- **Monocular depth is noisy.** Forward/back from torso scale jitters, hence
  zones and deadbands. A depth camera later would help.
- **Arm motion disturbs balance**, because the policies were trained with
  rigid arms. Measure it in sim first; C0 fixes it.
- **No ankle roll** limits lateral and single-leg mimicry. Set expectations.
- **The network path:** the Wi-Fi blocks direct device-to-device traffic, so
  it must go over Tailscale. A dropout must always end in "stand", which the
  timeouts guarantee.
- The 50 Hz loop must not gain work: overrun aborts are a known failure
  (fixed 10-22). Keep the network read O(1) and non-blocking.
