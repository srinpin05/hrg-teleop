# hrg-teleop

The HRG humanoid's arms and head copy what you do in front of your laptop camera.

```
LAPTOP (this repo)                                            ROBOT (Pi, ~/hrg_deploy)
camera -> MediaPipe Pose (local, ~10 ms) -> skeleton
  -> joint angles (shoulder pitch/roll, forearm twist, elbow, neck yaw/pitch)
  -> smoothing (One Euro filter)
  -> UDP ~30/s, newest wins, Tailscale  ----------------->   ./teleop.sh -> arms/arm_teleop.py
                                                             -> ArmBus (limits, slew) -> servos
```

All the heavy work runs on the laptop. The robot only receives about 400-byte
packets of joint angles. The legs aren't involved.

## Run it

**Once:**
1. Install Python 3, version 3.10 or newer, from python.org.
2. Install Tailscale on the laptop and log in to the same account as the
   robot. The robot's Wi-Fi network blocks devices from talking to each other
   directly, so Tailscale is the path that works: tested at 13 ms round trip.
3. Clone this repo.

**Every time:**
1. **Robot (Pi):** `cd ~/hrg_deploy && ./teleop.sh`. Use `./teleop.sh dry` to
   print without moving. Stop any other arm program first (`try_walk.sh`,
   `arms_only.sh`, `pose.sh`).
2. **Laptop:**
   ```bash
   ./run.sh            # Mac / Linux (first run sets itself up, about a minute)
   run.bat             # Windows
   ```
3. Stand back so the camera sees you **from the hips up**, then:

| key | does |
|---|---|
| **SPACE** | start/stop driving the robot. It starts **PAUSED**. |
| **C** | calibrate the head: look straight at the camera, press C |
| **M** | mirror on/off. **Mirror on (default):** facing the robot, your right arm moves its left arm, like a mirror. |
| **H** | head tracking on/off |
| **Q** / Esc | quit; the robot's arms go back to rest |

The window shows your skeleton, the robot's commanded pose (front and side),
every joint angle, and the status line. It shows **SENDING** with an `rtt` in
ms when the robot is answering, and "robot not replying" when it isn't.

Options:
| option | does |
|---|---|
| `--robot 100.97.141.88` | if the name `hrg` doesn't resolve. You can also set `HRG_ROBOT`. |
| `--dry` | camera and preview only, nothing sent. Use it to practise. |
| `--model lite` | faster model for a slow laptop (`heavy` = more accurate). Downloaded on first use. |
| `--camera 1` | a different webcam |
| `--smooth 0.8` | smoother, but laggier. `2.0` = snappier, but jittery. |
| `--no-mirror`, `--no-head` | start with those off |

On a Mac, the first run asks for camera permission for Terminal. Allow it,
then run again.

## Safety

- **Laptop:** nothing is sent until you press SPACE.
- **No person in view:** if nobody is visible for 0.5 s, the laptop tells the
  robot to rest.
- **Robot:**
  - It follows only while packets say "enabled" and are less than 0.3 s old;
    otherwise the arms ease back to rest. Wi-Fi drop, laptop closed or script
    crashed all end the same way.
  - Every angle is clamped to the servo limits and slew-limited to 3 rad/s
    (`ArmBus`).
  - `DEADMAN=4 ./teleop.sh` also requires the gamepad's LB to be held.
  - Ctrl+C sends the arms to rest and the servos go limp.

## Latency

| stage | typical |
|---|---|
| camera frame | 16–33 ms (30–60 fps) |
| pose model (`full`) | 9–12 ms measured on a desktop CPU; `lite` is faster |
| filter + mapping | < 1 ms |
| network (Tailscale) | ~7 ms one way (13 ms round trip measured) |
| robot loop | ≤ 20 ms (50 Hz) |
| servo travel | slew-capped at 3 rad/s; big fast moves trail a little |

Use `--smooth` to trade jitter against lag. The `rtt` on screen is the live
network part.

## How the mapping works (`hrg_teleop/retarget.py`)

1. **Your body frame.** It's built from your shoulders and hips: forward is
   the side your nose is on, left runs along your shoulders, up runs hips to
   shoulders. This makes it independent of where the camera is.
2. **Arm angles.** For each arm, the upper-arm and forearm directions in that
   frame become robot angles:
   - **Robot model:** shoulder_pitch → shoulder_roll → forearm_twist → elbow,
     checked against the robot's MuJoCo model.
   - **Signs:** pitch + forward, roll + outward, elbow + bend, twist + inward,
     on both arms.
   - **Shoulder choice:** the shoulder has two solutions, and it picks the one
     nearest the last command, so nothing flips 180°.
   - **Arm straight out sideways:** pitch is held, because the camera can't
     tell it apart there.
   - **Elbow nearly straight:** twist is held, because it's undefined there.
3. **Head.** Yaw and pitch come from the nose relative to your ears, minus
   the calibration (C).
4. **Mirror.** Swaps left and right and reflects the sideways components.

Tests: `python tests/test_retarget.py`. They check the forward model, an IK
round trip over 3000 random poses, no flip when raising the arm sideways,
mirror and same-side mode on a synthetic person, and that the camera frame's
handedness doesn't matter.

## Files

| file | what |
|---|---|
| `teleop.py` | main: camera, pose model, mapping, window, sending |
| `hrg_teleop/retarget.py` | landmarks → joint angles, plus the robot arm model and its limits |
| `hrg_teleop/filters.py` | One Euro filter |
| `hrg_teleop/net.py` | UDP link and round-trip measurement |
| `models/pose_landmarker_full.task` | MediaPipe model (Apache 2.0, see `models/NOTICE.md`) |
| `run.sh`, `run.bat` | one-command setup and run |

The robot side lives in the main robot code (desktop `~/hrg_urdf/deploy`, Pi
`~/hrg_deploy`): `teleop.sh` and `arms/arm_teleop.py`. The packet format is in
`hrg_teleop/net.py`.

## Roadmap

- [x] Arms and head, laptop camera, Tailscale
- [ ] Watch it in Foxglove at the same time (works already: ArmBus reports to the ROS view)
- [ ] Hand open/close (needs a gripper), wrist
- [ ] Legs: see "Adding legs" below

### Adding legs

The legs can't copy your joint angles directly. The robot balances with a
trained policy, and raw human leg angles would knock it over. Two steps:

1. **Commands, not angles.** Your movement becomes walk commands for the
   existing walk policy. Step toward the camera to walk forward, sideways to
   side-step, turn your shoulders to turn, stand still to stand.
   `run_policy.py` already takes vx/vy/wz from the gamepad, so it needs a
   network command input with the same dead-man and timeouts.
2. **Whole-body imitation.** Train a new policy in Isaac Lab that tracks a
   reference human pose: crouch, lean, lift a knee, step where you step. The
   method is the H2O / OmniH2O / HumanPlus line of work: retarget a human
   motion dataset (AMASS) to the HRG, train a tracking policy, then feed it
   this repo's skeleton live. Weeks of training and sim2real work.
