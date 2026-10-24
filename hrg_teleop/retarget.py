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

# Robot limits in degrees (arms/servo_calibration.json on the Pi, 2026-10-24),
# pulled in 2 deg so we never ask for the exact end stop. The robot clamps again.
LIMITS_DEG = {
    "right_shoulder_pitch": (-82, 178), "right_shoulder_roll": (0, 171),
    "right_forearm_twist": (-34, 88), "right_elbow": (0, 110),
    "left_shoulder_pitch": (-92, 174), "left_shoulder_roll": (0, 166),
    "left_forearm_twist": (-38, 84), "left_elbow": (0, 105),
    "neck_yaw": (-88, 88), "neck_pitch": (-30, 20),
}
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
    if prev is not None:
        ok.sort(key=lambda c: abs(c[0] - prev["shoulder_roll"]) + abs(c[1] - prev["shoulder_pitch"]))
    else:
        ok.sort(key=lambda c: c[0])            # prefer the smaller roll on the first frame
    roll, pitch = ok[0]
    if abs(math.cos(roll)) < lateral_cos:
        pitch = prev["shoulder_pitch"] if prev is not None else 0.0

    elbow = math.acos(max(-1.0, min(1.0, float(u @ f))))
    twist = prev["forearm_twist"] if prev is not None else 0.0
    if elbow > min_bend:
        d0 = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        fp = _unit(f - float(f @ u) * u)
        ang = math.atan2(float(u @ np.cross(d0, fp)), float(d0 @ fp))
        twist = s * ang
    return {
        "shoulder_pitch": clamp(f"{side}_shoulder_pitch", pitch),
        "shoulder_roll": clamp(f"{side}_shoulder_roll", roll),
        "forearm_twist": clamp(f"{side}_forearm_twist", twist),
        "elbow": clamp(f"{side}_elbow", elbow),
    }


# ---- whole frame ----
@dataclass
class Retargeter:
    mirror: bool = True          # facing the robot: your right arm drives ITS left, like a mirror
    head: bool = True
    min_visibility: float = 0.5
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
        if self.head and min(vis[NOSE], vis[L_EAR], vis[R_EAR]) >= self.min_visibility:
            yaw, pitch = self.head_angles(P, B)
            yaw -= self.head_neutral_yaw
            if self.mirror:
                yaw = -yaw                 # you look to your left -> it looks to ITS right
            out["neck_yaw"] = clamp("neck_yaw", yaw)
            out["neck_pitch"] = clamp("neck_pitch", pitch - self.head_neutral_pitch)
        return out

    @staticmethod
    def head_angles(P, B):
        ears = (P[L_EAR] + P[R_EAR]) / 2
        n = B @ (P[NOSE] - ears)
        yaw = math.atan2(n[1], n[0])
        pitch = math.atan2(n[2], math.hypot(n[0], n[1]))
        return yaw, pitch

    def calibrate_head(self, P):
        """Call while the person looks straight at the camera: that becomes neck 0."""
        yaw, pitch = self.head_angles(P, self.body_frame(P))
        self.head_neutral_yaw, self.head_neutral_pitch = yaw, pitch
