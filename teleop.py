"""HRG teleop: your laptop camera -> MediaPipe pose -> robot arm + head angles -> UDP to the robot.

    python teleop.py                     # sends to the robot "hrg" (Tailscale name)
    python teleop.py --robot 100.97.141.88
    python teleop.py --dry               # no robot: camera + skeleton + robot preview only

Keys (click the video window first):
    SPACE  start/stop sending to the robot. It starts PAUSED.
    C      calibrate: stand straight, look at the camera, press C (head, and legs if seen)
    M      mirror on/off. With mirror on (default), facing the robot, your
           right arm moves ITS left arm, like a mirror.
    H      head tracking on/off
    L      legs on/off (first draft: one-to-one, ONLY with the robot hanging in its
           frame). Needs both legs, hips to feet, in view. Robot: ./teleop.sh legs
    Q/ESC  quit (the robot's arms go to rest)

The robot side must be running:  cd ~/hrg_deploy && ./teleop.sh
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hrg_teleop.filters import JointFilter          # noqa: E402
from hrg_teleop.net import RobotLink                # noqa: E402
from hrg_teleop.retarget import Retargeter, arm_fk  # noqa: E402

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
LOST_AFTER = 0.5        # s without a person -> tell the robot to rest
SEND_HZ_PAUSED = 10     # heartbeat while paused, so the robot shows "link OK"

# skeleton lines for the overlay (MediaPipe Pose indices)
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (0, 7), (0, 8), (15, 19), (16, 20),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]
LEG_KEYS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")


def model_path(variant):
    p = os.path.join(HERE, "models", f"pose_landmarker_{variant}.task")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        print(f"downloading the {variant} pose model (one time)...")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), p)
    return p


def open_camera(index, w, h, fps):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)       # opens much faster on Windows
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                     # newest frame, not a queued old one
    return cap


def draw_robot(q, size=300):
    """Front view and side view of the robot's commanded arm/head pose."""
    img = np.full((size, size * 2, 3), 32, np.uint8)
    L = 0.15                                   # segment length, m (drawing only)
    k = size / 0.75                            # px per m

    def front(p, cx):     # viewer faces the robot: robot +y (its left) is screen RIGHT
        return int(cx + p[1] * k), int(size * 0.42 - p[2] * k)

    def side(p, cx):      # seen from the robot's left: forward (+x) is screen LEFT
        return int(cx - p[0] * k), int(size * 0.42 - p[2] * k)

    for view, cx, proj in (("front", size // 2, front), ("side (its left)", size + size // 2, side)):
        sh_y = 0.09
        torso = [np.array([0, sh_y, 0]), np.array([0, -sh_y, 0]),
                 np.array([0, -0.07, -0.22]), np.array([0, 0.07, -0.22])]
        pts = np.array([proj(p, cx) for p in torso], np.int32)
        cv2.polylines(img, [pts], True, (120, 120, 120), 2)
        for side_name, s, col in (("left", 1, (255, 170, 60)), ("right", -1, (40, 160, 255))):   # BGR: blue, orange
            g = lambda j: q.get(f"{side_name}_{j}", 0.0)
            u, f = arm_fk(s, g("shoulder_pitch"), g("shoulder_roll"), g("forearm_twist"), g("elbow"))
            a = np.array([0, s * sh_y, 0]); b = a + u * L; c = b + f * L
            cv2.line(img, proj(a, cx), proj(b, cx), col, 4)
            cv2.line(img, proj(b, cx), proj(c, cx), col, 4)
        for side_name, s, col in (("left", 1, (255, 170, 60)), ("right", -1, (40, 160, 255))):
            if f"{side_name}_knee" not in q:
                continue
            g = lambda j: q.get(f"{side_name}_{j}", 0.0)
            p, r, kn = g("hip_pitch"), g("hip_roll"), g("knee")
            hip = np.array([0, s * 0.05, -0.24])
            t = np.array([math.sin(p), s * math.sin(r) * math.cos(p), -math.cos(r) * math.cos(p)])
            sh = np.array([math.sin(p - kn), s * math.sin(r) * math.cos(p - kn), -math.cos(r) * math.cos(p - kn)])
            k_ = hip + t * 0.1; a_ = k_ + sh * 0.1
            cv2.line(img, proj(hip, cx), proj(k_, cx), col, 3)
            cv2.line(img, proj(k_, cx), proj(a_, cx), col, 3)
        head = np.array([0, 0, 0.08])
        cv2.circle(img, proj(head, cx), int(0.05 * k), (200, 200, 200), 2)
        yaw, pitch = q.get("neck_yaw", 0.0), q.get("neck_pitch", 0.0)
        nose = head + 0.07 * np.array([math.cos(pitch) * math.cos(yaw),
                                       math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        cv2.line(img, proj(head, cx), proj(nose, cx), (0, 255, 0), 3)
        cv2.putText(img, view, (cx - 60, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(img, "ROBOT (commanded)  orange=its right  blue=its left", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=os.environ.get("HRG_ROBOT", "hrg"),
                    help="robot host/IP (default hrg; or 100.97.141.88 on Tailscale, or its Wi-Fi IP)")
    ap.add_argument("--port", type=int, default=9874)
    ap.add_argument("--dry", action="store_true", help="don't send anything")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="use a video file instead of the camera (testing)")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"],
                    help="lite = fastest, heavy = most accurate (default full)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360,
                    help="360 not 480: a Logitech C270 does 30 fps at 640x360 but only 15 at 640x480")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--legs", action="store_true", help="start with legs on (L toggles). Robot must be hanging.")
    ap.add_argument("--leg-feet", action="store_true", help="follow your foot angle too (default: robot feet flat)")
    ap.add_argument("--smooth", type=float, default=1.2,
                    help="filter min cutoff Hz: lower = smoother but laggier (default 1.2)")
    ap.add_argument("--record", help="write every frame (landmarks, raw and filtered angles) to this .jsonl")
    ap.add_argument("--snap", help="save the window image every 10th frame into this folder")
    ap.add_argument("--autostart", type=float, default=-1, metavar="S",
                    help="start SENDING by itself S seconds after the camera opens (no SPACE needed)")
    ap.add_argument("--max-frames", type=int, default=0, help="quit after N frames (testing)")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path(args.model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5))

    cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit("can't open the camera. On a Mac, allow camera access for the Terminal app "
                         "(System Settings > Privacy & Security > Camera); try --camera 1")
    link = RobotLink(args.robot, args.port, dry=args.dry)
    rt = Retargeter(mirror=not args.no_mirror, head=not args.no_head, legs=args.legs, leg_feet=args.leg_feet)
    last_vis = None
    # beta 1.0 (was 0.4): the filter opens up faster on quick moves -> less lag
    filt = JointFilter(min_cutoff=args.smooth, beta=1.0)
    legs_seen = 0.0
    sending = False
    cmd = {}                     # last command per joint (joints not seen this frame hold)
    last_seen = 0.0
    last_paused_send = 0.0
    t0 = time.monotonic()
    fps = infer_ms = 0.0
    n = 0
    last_P = None
    rec = open(args.record, "w", buffering=1) if args.record else None
    if args.snap:
        os.makedirs(args.snap, exist_ok=True)
    t_auto = time.monotonic() + args.autostart if args.autostart >= 0 and not link.dry else None
    print(f"robot {args.robot}:{args.port} ({'DRY RUN' if args.dry else link.addr or link.error}) -- "
          "SPACE = start/stop sending, C = calibrate head, M = mirror, Q = quit")
    while True:
        t_frame = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if t_auto is not None and time.monotonic() >= t_auto:
            sending, t_auto = True, None
            print("SENDING (autostart)")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ti = time.monotonic()
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int((ti - t0) * 1000))
        infer_ms = 0.9 * infer_ms + 0.1 * (time.monotonic() - ti) * 1000
        now = time.monotonic()
        person = bool(res.pose_world_landmarks)
        if person:
            W = res.pose_world_landmarks[0]
            P = np.array([[p.x, p.y, p.z] for p in W])
            vis = np.array([p.visibility if p.visibility is not None else 1.0 for p in W])
            last_P, last_vis = P, vis
            q = rt.solve(P, vis)
            if q:
                last_seen = now
                if any(k.endswith("_knee") for k in q):
                    legs_seen = now
                cmd.update(filt(q, now))
            if rec:
                import json
                rec.write(json.dumps({"t": round(now - t0, 3), "P": np.round(P, 3).tolist(),
                                      "vis": np.round(vis, 2).tolist(),
                                      "q": {k: round(v, 4) for k, v in q.items()},
                                      "cmd": {k: round(v, 4) for k, v in cmd.items()},
                                      "sending": sending}) + "\n")
        tracking = now - last_seen < LOST_AFTER
        if sending and tracking:
            arms = {k: v for k, v in cmd.items() if not k.endswith(LEG_KEYS)}
            legs = ({k: v for k, v in cmd.items() if k.endswith(LEG_KEYS)}
                    if rt.legs and now - legs_seen < 0.3 else None)
            link.send(True, arms, legs)
        elif now - last_paused_send > 1.0 / SEND_HZ_PAUSED:
            link.send(False, {})                 # robot: ease to rest / stay at rest
            last_paused_send = now

        # ---- display ----
        dt = now - t_frame
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
        if not args.headless:
            view = cv2.flip(frame, 1)            # selfie view; the model saw the unflipped frame
            h, w = view.shape[:2]
            if person:
                pts = [(int((1 - p.x) * w), int(p.y * h)) for p in res.pose_landmarks[0]]
                for a, b in BONES:
                    cv2.line(view, pts[a], pts[b], (0, 255, 0), 2)
                for i in (0, 11, 12, 13, 14, 15, 16):
                    cv2.circle(view, pts[i], 4, (0, 0, 255), -1)
            if link.dry:
                state, col = "DRY RUN (not sending)", (200, 200, 200)
            elif not sending and t_auto is not None:
                state, col = f"GET IN POSITION - robot starts in {t_auto - now:.0f} s", (0, 200, 255)
            elif not sending:
                state, col = "PAUSED - press SPACE to drive the robot", (0, 200, 255)
            elif not tracking:
                state, col = "NO PERSON - robot resting", (0, 0, 255)
            else:
                state, col = "SENDING TO ROBOT", (0, 255, 0)
            replied = link.last_ack and now - link.last_ack < 1.5
            net = ("dry" if link.dry else link.error or
                   (f"rtt {link.rtt_ms:.0f} ms" if replied and link.rtt_ms is not None else "robot not replying"))
            lines = [state,
                     f"{fps:4.0f} fps  model {infer_ms:4.0f} ms  {net}",
                     f"mirror {'ON' if rt.mirror else 'off'}  head {'ON' if rt.head else 'off'}  legs "
                     + (("ON (seen)" if now - legs_seen < 0.3 else "ON - NEED BOTH LEGS IN VIEW") if rt.legs else "off")]
            for i, txt in enumerate(lines):
                cv2.putText(view, txt, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            col if i == 0 else (255, 255, 255), 2)
            panel = draw_robot(cmd, size=h // 2)
            joints = np.full((h - panel.shape[0], panel.shape[1], 3), 24, np.uint8)
            arm_items = sorted((k, v) for k, v in cmd.items() if not k.endswith(LEG_KEYS))
            leg_items = sorted((k, v) for k, v in cmd.items() if k.endswith(LEG_KEYS)) if rt.legs else []
            colw = joints.shape[1] // 2
            for c, items in enumerate((arm_items, leg_items)):
                for i, (k, v) in enumerate(items):
                    name = k.replace("shoulder_", "sh_").replace("forearm_", "fa_").replace("_pitch", "_p")
                    cv2.putText(joints, f"{name:15s}{math.degrees(v):+5.0f}", (6 + c * colw, 14 + 13 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1)
            full = np.hstack([view, np.vstack([panel, joints])])
            if args.snap and n % 10 == 0:
                cv2.imwrite(os.path.join(args.snap, f"f{n:05d}.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 70])
            cv2.imshow("HRG teleop", full)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                sending = not sending and not link.dry
                print("SENDING" if sending else "paused")
            elif key == ord("m"):
                rt.mirror = not rt.mirror
                rt.prev.clear()
            elif key == ord("l"):
                rt.legs = not rt.legs
                if not rt.legs:
                    for k in [k for k in cmd if k.endswith(LEG_KEYS)]:
                        del cmd[k]
                print("legs", "ON" if rt.legs else "off")
            elif key == ord("h"):
                rt.head = not rt.head
            elif key == ord("c") and last_P is not None:
                rt.calibrate_head(last_P)
                legs_ok = rt.calibrate_legs(last_P, last_vis)
                print("head calibrated" + ("; legs calibrated (standing straight = 0)" if legs_ok else "; legs not in view"))
        if args.max_frames and n >= args.max_frames:
            break
    for _ in range(3):
        link.send(False, {})
    if rec:
        rec.close()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print(f"{n} frames, {fps:.0f} fps, model {infer_ms:.0f} ms")


if __name__ == "__main__":
    main()
