"""One Euro filter (Casiez et al. 2012): smooths jitter when you hold still,
and barely lags when you move fast. It's the standard filter for pose
tracking, because a plain low-pass trades one for the other."""
from __future__ import annotations

import math


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.4, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class JointFilter:
    """One OneEuro per joint name."""

    def __init__(self, **kw):
        self.kw = kw
        self.f = {}

    def __call__(self, q, t):
        out = {}
        for k, v in q.items():
            if k not in self.f:
                self.f[k] = OneEuro(**self.kw)
            out[k] = self.f[k](v, t)
        return out
