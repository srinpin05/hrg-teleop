"""Human pose (MediaPipe world landmarks) -> HRG arm and neck joint angles.

ROBOT ARM MODEL (verified against sim/hrg.xml on 2026-10-24). Body frame:
x forward, y left, z up. Each arm is shoulder_pitch -> shoulder_roll ->
forearm_twist -> elbow:

    upper-arm direction  u = Ry(-pitch) Rx(s*roll) (0, 0, -1)
                           = (sin p cos r,  s sin r,  -cos p cos r)
    elbow bend direction d0 = Ry(-pitch) (1, 0, 0) = (cos p, 0, sin p)   at twist 0
                         d = rotate d0 about u by s*twist
    forearm direction    f = cos(elbow) u + sin(elbow) d

where s = +1 for the left arm and -1 for the right. Signs:
- pitch + = arm forward/up
- roll + = arm out to the side
- elbow + = bend
- twist + = inward rotation, so a bent forearm swings toward the belly or face
- neck_yaw + = look left
- neck_pitch + = look up

All angles here are radians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# MediaPipe Pose landmark indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE = 25, 26, 27, 28, 29, 30, 31, 32

# Robot limits in degrees (arms/servo_calibration.json on the Pi, 2026-10-24),
# pulled in 2 deg so we never ask for the exact end stop. The robot clamps again.
LIMITS_DEG = {
    "right_shoulder_pitch": (-82, 178), "right_shoulder_roll": (0, 171),
    "right_forearm_twist": (-34, 88), "right_elbow": (0, 110),
    "left_shoulder_pitch": (-92, 174), "left_shoulder_roll": (0, 166),
    "left_forearm_twist": (-38, 84), "left_elbow": (0, 105),
    "neck_yaw": (-88, 88), "neck_pitch": (-30, 20),
}
# LEGS, first draft (2026-10-25), for the robot hanging in its frame: one to one,
# no balance. Signs from sim/hrg.xml (read only): hip_pitch + = thigh forward,
# hip_roll + = leg outward, knee + = bend, ankle_pitch + = toes up. 0 = straight
# leg. Ranges are deliberately narrower than the URDF; the robot clamps again.
# hip_yaw is held at 0 (a webcam can't see thigh rotation reliably).
LEG_LIMITS_DEG = {
    "hip_yaw": (0, 0), "hip_roll": (-15, 35), "hip_pitch": (-12, 85),
    "knee": (0, 95), "ankle_pitch": (-40, 45),
}
for _side in ("left", "right"):
    for _j, _r in LEG_LIMITS_DEG.items():
        LIMITS_DEG[f"{_side}_{_j}"] = _r
LIMITS = {k: (math.radians(a), math.radians(b)) for k, (a, b) in LIMITS_DEG.items()}
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "forearm_twist", "elbow")


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def clamp(name, rad):
    lo, hi = LIMITS[name]
    return min(hi, max(lo, rad))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot_about(axis, v, ang):
    """Rodrigues: rotate v about unit axis by ang."""
    axis = _unit(axis)
    return (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
            + axis * float(axis @ v) * (1 - math.cos(ang)))


# ---- forward model (tests, and the robot preview drawn on the laptop) ----
def arm_fk(s, pitch, roll, twist, elbow):
    """-> (upper-arm unit vector, forearm unit vector) in the body frame."""
    sp, cp, sr, cr = math.sin(pitch), math.cos(pitch), math.sin(roll), math.cos(roll)
    u = np.array([sp * cr, s * sr, -cp * cr])
    d0 = np.array([cp, 0.0, sp])
    d = rot_about(u, d0, s * twist)
    f = math.cos(elbow) * u + math.sin(elbow) * d
    return u, _unit(f)


# ---- inverse: vectors -> angles ----
def arm_ik(s, u, f, prev=None, min_bend=math.radians(35), lateral_cos=0.25):
    """Upper-arm u and forearm f (body frame) -> dict of the 4 angles.

    The shoulder has two solutions, (roll, pitch) and (pi - roll, pitch + pi).
    We keep the one nearest the previous command, which avoids a 180° pitch
    flip when the arm passes through straight-out-sideways. Close to straight
    out sideways (|cos roll| < lateral_cos, within about 15°), pitch hardly
    changes where the arm points and the camera noise would swing it wildly,
    so we hold the previous pitch (0 on the first frame) and only update roll.
    Twist is undefined when the elbow is nearly straight; then we hold the
    previous twist.
    """
    u, f = _unit(np.asarray(u, float)), _unit(np.asarray(f, float))
    r0 = math.asin(max(-1.0, min(1.0, s * u[1])))
    r0 = max(0.0, r0)                                   # roll can't go inward of the body
    cands = []
    for roll, flip in ((r0, False), (math.pi - r0, True)):
        p = math.atan2(u[0], -u[2])
        if flip:
            p = wrap(p + math.pi)
        cands.append((roll, p))
    side = "left" if s > 0 else "right"
    ok = []
    for roll, p in cands:
        lo_p, hi_p = LIMITS[f"{side}_shoulder_pitch"]
        lo_r, hi_r = LIMITS[f"{side}_shoulder_roll"]
        # also try p +/- 2pi so a +170 deg reach isn't rejected as -190
        for pp in (p, p + 2 * math.pi, p - 2 * math.pi):
            if lo_p - 0.2 <= pp <= hi_p + 0.2 and lo_r - 0.05 <= roll <= hi_r + 0.05:
                ok.append((roll, pp))
    if not ok:
        ok = [cands[0]]
    if prev is not None and u[2] > -0.2:
        # arm level or raised: the branch is genuinely ambiguous, stay continuous
        ok.sort(key=lambda c: abs(c[0] - prev["shoulder_roll"]) + abs(c[1] - prev["shoulder_pitch"]))
    else:
        # arm pointing down: only the small-roll branch is natural. Resetting to it
        # here means a fast move that landed on the flipped branch recovers.
        ok.sort(key=lambda c: c[0])
    roll, pitch = ok[0]
    if prev is not None:
        # near straight-out-sideways pitch barely moves the arm and camera noise
        # swings it: blend from "hold" (|cos roll| <= 0.15) to "follow" (>= 0.5)
        w = min(1.0, max(0.0, (abs(math.cos(roll)) - lateral_cos * 0.6) / (0.5 - lateral_cos * 0.6)))
        pitch = prev["shoulder_pitch"] + w * wrap(pitch - prev["shoulder_pitch"])
    elif abs(math.cos(roll)) < lateral_cos:
        pitch = 0.0

    elbow = math.acos(max(-1.0, min(1.0, float(u @ f))))
    twist = prev["forearm_twist"] if prev is not None else 0.0
    if elbow > min_bend * 0.55:
        d0 = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        fp = _unit(f - float(f @ u) * u)
        ang = math.atan2(float(u @ np.cross(d0, fp)), float(d0 @ fp))
        # twist is undefined with a straight elbow: blend in between ~20 and 35 deg
        w = min(1.0, (elbow - min_bend * 0.55) / (min_bend * 0.45))
        twist = twist + w * (s * ang - twist)
    return {
        "shoulder_pitch": clamp(f"{side}_shoulder_pitch", pitch),
        "shoulder_roll": clamp(f"{side}_shoulder_roll", roll),
        "forearm_twist": clamp(f"{side}_forearm_twist", twist),
        "elbow": clamp(f"{side}_elbow", elbow),
    }


def leg_ik(s, t, sh, foot=None, off=None):
    """Thigh t, shin sh, foot heel->toe (body frame, unit or not) -> leg angles.
    Hip modelled as roll (outer) then pitch: t = (sin p, s sin r cos p, -cos r cos p).
    Foot: robot toe-up angle = hip_pitch - knee + ankle, so ankle = phi - hip + knee
    where phi is the human foot's toe-up angle (0 = flat when foot not seen)."""
    side = "left" if s > 0 else "right"
    t, sh = _unit(np.asarray(t, float)), _unit(np.asarray(sh, float))
    pitch = math.asin(max(-1.0, min(1.0, t[0])))
    roll = math.atan2(s * t[1], -t[2])
    knee = math.acos(max(-1.0, min(1.0, float(t @ sh))))
    if off:     # standing-straight calibration (key C): the camera reads ~20 deg of knee when straight
        pitch -= off.get("hip_pitch", 0.0)
        roll -= off.get("hip_roll", 0.0)
        knee = max(0.0, knee - off.get("knee", 0.0))
    phi = 0.0
    if foot is not None:
        fv = np.asarray(foot, float)
        phi = max(-0.5, min(0.5, math.atan2(fv[2], math.hypot(fv[0], fv[1]))))
    ankle = phi - pitch + knee
    return {f"{side}_hip_yaw": 0.0,
            f"{side}_hip_roll": clamp(f"{side}_hip_roll", roll),
            f"{side}_hip_pitch": clamp(f"{side}_hip_pitch", pitch),
            f"{side}_knee": clamp(f"{side}_knee", knee),
            f"{side}_ankle_pitch": clamp(f"{side}_ankle_pitch", ankle)}


# ---- whole frame ----
@dataclass
class Retargeter:
    mirror: bool = True          # facing the robot: your right arm drives ITS left, like a mirror
    head: bool = True
    min_visibility: float = 0.3      # was 0.5: motion blur on fast moves froze the arm
    legs: bool = False
    leg_feet: bool = False        # follow the foot angle; off = robot feet stay flat (foot landmarks are noisy)
    leg_off: dict = field(default_factory=dict)    # side -> {hip_pitch, hip_roll, knee} from calibrate_legs()
    head_neutral_pitch: float = 0.0     # rad, set by calibrate_head()
    head_neutral_yaw: float = 0.0
    prev: dict = field(default_factory=dict)

    def body_frame(self, P):
        """3x3 rows = body x (forward), y (left), z (up) in landmark coordinates."""
        mid_sh = (P[L_SH] + P[R_SH]) / 2
        mid_hip = (P[L_HIP] + P[R_HIP]) / 2
        y = _unit(P[L_SH] - P[R_SH])                     # the person's left
        z = _unit(mid_sh - mid_hip)
        z = _unit(z - float(z @ y) * y)
        x = _unit(np.cross(y, z))
        # MediaPipe's world-frame handedness doesn't matter: forward is the side the nose is on
        ears = (P[L_EAR] + P[R_EAR]) / 2
        if float(x @ (P[NOSE] - ears)) < 0:
            x = -x
        return np.vstack([x, y, z])

    def solve(self, P, vis):
        """P: (33, 3) world landmarks, vis: (33,) visibility. -> {joint: rad} (only joints seen)."""
        if min(vis[L_SH], vis[R_SH], vis[L_HIP], vis[R_HIP]) < self.min_visibility * 0.6:
            return {}
        B = self.body_frame(P)
        out = {}
        # (robot side, sign, human shoulder/elbow/wrist)
        if self.mirror:
            arms = (("left", 1, R_SH, R_EL, R_WR), ("right", -1, L_SH, L_EL, L_WR))
        else:
            arms = (("left", 1, L_SH, L_EL, L_WR), ("right", -1, R_SH, R_EL, R_WR))
        for side, s, a, b, c in arms:
            if min(vis[a], vis[b], vis[c]) < self.min_visibility:
                continue
            u = B @ (P[b] - P[a])
            f = B @ (P[c] - P[b])
            if self.mirror:                # reflect across the sagittal plane
                u[1], f[1] = -u[1], -f[1]
            prev = self.prev.get(side)
            q = arm_ik(s, u, f, prev)
            self.prev[side] = q
            for j, v in q.items():
                out[f"{side}_{j}"] = v
        if self.legs:
            out.update(self.solve_legs(P, vis, B))
        if self.head and min(vis[NOSE], vis[L_EAR], vis[R_EAR]) >= self.min_visibility:
            yaw, pitch = self.head_angles(P, B)
            yaw -= self.head_neutral_yaw
            if self.mirror:
                yaw = -yaw                 # you look to your left -> it looks to ITS right
            out["neck_yaw"] = clamp("neck_yaw", yaw)
            out["neck_pitch"] = clamp("neck_pitch", pitch - self.head_neutral_pitch)
        return out

    def solve_legs(self, P, vis, B, min_vis=0.5, raw=False):
        """Both legs or nothing: hips, knees and ankles of both legs must be seen."""
        if min(vis[i] for i in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)) < min_vis:
            return {}
        # leg frame: up = TRUE vertical (camera's up; MediaPipe world y points down),
        # NOT the torso -- bending over the desk must not lift the robot's legs.
        # y along the hips (levelled), forward agrees with the body frame B.
        z = np.array([0.0, -1.0, 0.0])
        if float(z @ B[2]) < 0:
            z = -z
        y = _unit(P[L_HIP] - P[R_HIP])
        y = _unit(y - float(y @ z) * z)
        x = _unit(np.cross(y, z))
        if float(x @ B[0]) < 0:
            x = -x
        H = np.vstack([x, y, z])
        if self.mirror:
            legs = (("left", 1, R_HIP, R_KNEE, R_ANK, R_HEEL, R_TOE),
                    ("right", -1, L_HIP, L_KNEE, L_ANK, L_HEEL, L_TOE))
        else:
            legs = (("left", 1, L_HIP, L_KNEE, L_ANK, L_HEEL, L_TOE),
                    ("right", -1, R_HIP, R_KNEE, R_ANK, R_HEEL, R_TOE))
        out = {}
        for side, s, h, k, a, he, to in legs:
            t = H @ (P[k] - P[h])
            sh = H @ (P[a] - P[k])
            foot = (H @ (P[to] - P[he]) if self.leg_feet and not raw
                    and min(vis[he], vis[to]) >= min_vis else None)
            if self.mirror:
                t[1], sh[1] = -t[1], -sh[1]
            out.update(leg_ik(s, t, sh, foot, None if raw else self.leg_off.get(side)))
        return out

    @staticmethod
    def head_angles(P, B):
        ears = (P[L_EAR] + P[R_EAR]) / 2
        n = B @ (P[NOSE] - ears)
        yaw = math.atan2(n[1], n[0])
        pitch = math.atan2(n[2], math.hypot(n[0], n[1]))
        return yaw, pitch

    def calibrate_legs(self, P, vis):
        """Call while the person stands straight: that becomes legs straight (all 0). -> True if done."""
        B = self.body_frame(P)
        q = self.solve_legs(P, vis, B, raw=True)
        if not q:
            return False
        # only accept an upright, roughly straight stance (torso within 28 deg of camera vertical,
        # knees < 40, hips < 30): calibrating on a bent pose made "straight" read bent
        up = abs(B[2][1])          # |torso up . camera vertical|
        if up < math.cos(math.radians(28)) or max(abs(q[f"{s}_{j}"]) for s in ("left", "right")
                                                 for j in ("knee", "hip_pitch")) > math.radians(40) \
                or max(abs(q["left_hip_pitch"]), abs(q["right_hip_pitch"])) > math.radians(30):
            return False
        for side in ("left", "right"):
            self.leg_off[side] = {j: q[f"{side}_{j}"] for j in ("hip_pitch", "hip_roll", "knee")}
        return True

    def calibrate_head(self, P):
        """Call while the person looks straight at the camera: that becomes neck 0."""
        yaw, pitch = self.head_angles(P, self.body_frame(P))
        self.head_neutral_yaw, self.head_neutral_pitch = yaw, pitch
