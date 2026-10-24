# Plan: make the arm and head mimic precise

Status: **PLAN, not implemented** (2026-10-24). Seth's report after the first
live run: "it works, but the motor movements can be more precise and
accurate... if the hand or the head moves too fast, the robot's hand doesn't
know how to move to the end position."

## Likely causes, from the current code (to confirm with data, step 1)

| # | cause | where | effect |
|---|---|---|---|
| 1 | **Joints freeze when a landmark's visibility drops below 0.5.** Motion blur on a fast move does exactly that, so the command sticks at the last sharp frame. | `retarget.py` `Retargeter.solve` (`min_visibility`) | arm stops partway and may not resume until you're fully still and sharp again |
| 2 | **Shoulder branch "nearest previous".** A fast move can land on the flipped solution (roll > 90°, pitch + 180°), which then clamps at a joint limit and reaches a different end pose. | `arm_ik` | wrong end position after a fast move |
| 3 | **Held values:** pitch held within ~15° of straight-sideways; twist held while the elbow is < 35°. A fast move through those zones leaves a stale value. | `arm_ik` | wrong pitch or twist at the end |
| 4 | **Servo slew caps:** 3 rad/s, left_shoulder_roll 1.2 rad/s. A human arm moves 5–10 rad/s. | Pi `arm_bus.py` | lag; it arrives late (should still arrive) |
| 5 | **Smoothing** (One Euro, min_cutoff 1.2) | `filters.py` | small extra lag |
| 6 | **Monocular depth noise:** MediaPipe world z | camera | forward/back (pitch) imprecise, jitter |
| 7 | **The robot's arm geometry vs the model.** In the MuJoCo model the RIGHT arm at all-zeros hangs about 30° outward and the left about 10°, while the code assumes 0 = straight down. | `servo_calibration.json` / URDF | constant offset on one arm |
| 8 | **Servo zero and scale** (`center_us`, `us_per_rad`) error and hobby-servo deadband | Pi calibration | static error even with perfect tracking |
| 9 | **15 fps camera:** a C270 at 640×480 (fixed: 640×360 gives 30 fps). Low light also blurs. | camera | blur → cause 1 |

## Steps

1. **Measure before changing anything:**
   - **Laptop:** `--record FILE` logs, per frame, raw world and image
     landmarks with visibility, IK output, the filtered and sent command,
     and timestamps.
   - **Pi:** `arm_teleop.py` logs received, commanded (after slew) and
     timestamps.
   - **Replay tool:** re-run retarget offline on a recording, so every fix is
     tested on the same motion.
   - **Record:** 10 slow and 10 fast moves to fixed end poses (arm
     forward/up/side/across chest, head turns). Film the robot plus the
     person at 120–240 fps with a phone for real end-to-end latency and
     end-pose error.
2. **Fix tracking dropouts (cause 1):**
   - Replace freeze-on-low-visibility with confidence-weighted updates:
     - accept down to about 0.3 visibility, with more smoothing
     - for short gaps (< 0.3 s), extrapolate the joint velocity with decay
     - after the gap, **converge to the fresh pose**
   - Always re-solve from the current landmarks when the person stops, so
     the end pose is reached.
3. **Fix branch and holds (causes 2 and 3):**
   - In the unambiguous region (|cos roll| > 0.5) **always reset to the
     canonical branch** (roll ≤ 90°); the nearest-previous rule applies only
     near the singularity.
   - Score branches by joint distance **plus limit violation**.
   - Replace the hard pitch hold with a weight that blends towards the
     computed pitch as |cos roll| grows.
   - For twist, blend in by elbow angle (20–45°) instead of a hard 35°
     cut-off.
   - Add tests: fast sweeps, a sideways crossing, across-chest poses.
4. **Speed (causes 4 and 5):**
   - Measure each servo's real top speed at the arm load
     (`arm_zero.py --goto` big steps, time them).
   - Raise the teleop slew cap to about 80% of it (per joint in
     `servo_map.JOINT_SLEW`). Keep left_shoulder_roll limited unless the
     hunting is fixed.
   - Add **latency compensation:** send command plus velocity × (measured
     latency), capped.
   - Re-tune the filter from recordings: lowest lag with acceptable jitter.
5. **Static accuracy (causes 7 and 8):**
   - On the robot, command known poses (0, 45, 90° per joint) and measure
     with a protractor or photo. Fix `center_us` / `us_per_rad` /
     zero offsets via `arm_zero.py`.
   - Check the right-arm offset question (cause 7) and correct either the
     model or the calibration so that 0 = arm straight down.
6. **Person-specific calibration:** a T-pose and arms-down capture (key
   `T`) that sets per-person arm neutral and limits, like `C` does for the
   head.
7. **Model and camera:**
   - Compare `--model heavy` and `full` on the recordings (accuracy vs ms).
   - Good front lighting.
   - Optional 60 fps camera.
8. **Gate:**
   - End-pose error < 8° per joint on the 20 test moves, including fast ones.
   - The robot reaches the end pose within 0.5 s of the person stopping.
   - No 180° flips.
   - Latency (person → servo start) < 200 ms on video.
