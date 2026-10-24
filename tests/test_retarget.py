"""Run: python -m pytest tests  (or python tests/test_retarget.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrg_teleop.retarget import (L_EAR, L_EL, L_HIP, L_SH, L_WR, NOSE, R_EAR, R_EL,  # noqa: E402
                                 R_HIP, R_SH, R_WR, LIMITS, Retargeter, arm_fk, arm_ik)

D = math.radians


def test_fk_known_poses():
    u, f = arm_fk(1, 0, 0, 0, 0)
    assert np.allclose(u, [0, 0, -1]) and np.allclose(f, [0, 0, -1])        # hanging
    u, _ = arm_fk(1, D(90), 0, 0, 0)
    assert np.allclose(u, [1, 0, 0], atol=1e-9)                               # pitch + = forward
    u, _ = arm_fk(1, 0, D(90), 0, 0)
    assert np.allclose(u, [0, 1, 0], atol=1e-9)                               # left roll + = out (+y)
    u, _ = arm_fk(-1, 0, D(90), 0, 0)
    assert np.allclose(u, [0, -1, 0], atol=1e-9)                              # right roll + = out (-y)
    _, f = arm_fk(1, 0, 0, 0, D(90))
    assert np.allclose(f, [1, 0, 0], atol=1e-9)                               # elbow bends forward
    _, f = arm_fk(1, 0, 0, D(45), D(90))
    assert f[1] < -0.5                                                         # left twist + = inward (-y)
    _, f = arm_fk(-1, 0, 0, D(45), D(90))
    assert f[1] > 0.5                                                          # right twist + = inward (+y)


def test_ik_round_trip():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3000):
        s = rng.choice([1, -1])
        side = "left" if s > 0 else "right"
        lo = {j: LIMITS[f"{side}_{j}"] for j in ("shoulder_pitch", "shoulder_roll", "forearm_twist", "elbow")}
        q = {j: rng.uniform(a + 0.05, b - 0.05) for j, (a, b) in lo.items()}
        q["shoulder_roll"] = rng.uniform(0.05, D(85))      # unambiguous branch
        q["elbow"] = rng.uniform(D(35), lo["elbow"][1] - 0.05)
        u, f = arm_fk(s, q["shoulder_pitch"], q["shoulder_roll"], q["forearm_twist"], q["elbow"])
        r = arm_ik(s, u, f, prev=q)
        u2, f2 = arm_fk(s, r["shoulder_pitch"], r["shoulder_roll"], r["forearm_twist"], r["elbow"])
        worst = max(worst, float(np.linalg.norm(u - u2)), float(np.linalg.norm(f - f2)))
    assert worst < 1e-6, worst


def test_no_flip_through_sideways():
    """Raising the arm sideways past horizontal must not flip pitch by 180 deg."""
    prev = None
    for deg in range(0, 165, 3):
        u, f = arm_fk(1, 0, D(deg), 0, 0)
        q = arm_ik(1, u, f, prev)
        assert abs(q["shoulder_pitch"]) < D(5), (deg, q)
        prev = q


def make_person(left_arm, right_arm, head_yaw=0.0):
    """Synthetic MediaPipe world landmarks: x = person's left->right flipped like an image
    (x to the subject's right), y down, z toward camera; the person faces the camera (-z).
    left_arm/right_arm: (pitch, roll, twist, elbow) in the ROBOT convention."""
    P = np.zeros((33, 3))
    # body frame (fwd, left, up) -> mediapipe (x right-of-subject... we use: x = -left, y = -up, z = -fwd)
    M = np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]], float).T
    to_mp = lambda v: M.T @ np.asarray(v, float)

    def put(i, v):
        P[i] = to_mp(v)
    put(L_SH, [0, 0.18, 0.5]); put(R_SH, [0, -0.18, 0.5])
    put(L_HIP, [0, 0.1, 0]); put(R_HIP, [0, -0.1, 0])
    for s, sh, el, wr, q in ((1, L_SH, L_EL, L_WR, left_arm), (-1, R_SH, R_EL, R_WR, right_arm)):
        u, f = arm_fk(s, *q)
        e = np.array([0, s * 0.18, 0.5]) + 0.28 * u
        put(el, e); put(wr, e + 0.25 * f)
    head = np.array([0, 0, 0.65])
    put(L_EAR, head + [0, 0.07, 0]); put(R_EAR, head + [0, -0.07, 0])
    put(NOSE, head + 0.1 * np.array([math.cos(head_yaw), math.sin(head_yaw), 0]))
    # rotate the ears with the head
    for i, sgn in ((L_EAR, 1), (R_EAR, -1)):
        put(i, head + sgn * 0.07 * np.array([-math.sin(head_yaw), math.cos(head_yaw), 0]))
    return P, np.ones(33)


def test_person_same_side_and_mirror():
    left = (D(60), D(20), D(30), D(70))
    right = (D(-20), D(45), D(0), D(10))
    P, vis = make_person(left, right, head_yaw=D(30))
    q = Retargeter(mirror=False).solve(P, vis)
    assert abs(q["left_shoulder_pitch"] - left[0]) < 1e-6 and abs(q["left_elbow"] - left[3]) < 1e-6
    assert abs(q["right_shoulder_roll"] - right[1]) < 1e-6
    assert abs(q["neck_yaw"] - D(30)) < 1e-6
    m = Retargeter(mirror=True).solve(P, vis)
    # mirror: the person's LEFT arm drives the robot's RIGHT arm, same angles
    assert abs(m["right_shoulder_pitch"] - left[0]) < 1e-6 and abs(m["right_elbow"] - left[3]) < 1e-6
    assert abs(m["right_forearm_twist"] - left[2]) < 1e-6
    assert abs(m["left_shoulder_roll"] - right[1]) < 1e-6
    assert abs(m["neck_yaw"] + D(30)) < 1e-6


def test_landmark_handedness_does_not_matter():
    """MediaPipe's world axes may be left- or right-handed: answers must not change."""
    P, vis = make_person((D(60), D(20), D(30), D(70)), (D(-20), D(45), 0, D(10)), D(30))
    a = Retargeter(mirror=False).solve(P, vis)
    P2 = P.copy(); P2[:, 0] *= -1
    b = Retargeter(mirror=False).solve(P2, vis)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
