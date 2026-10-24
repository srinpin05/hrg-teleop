"""Laptop -> robot UDP link. Fire-and-forget, newest packet wins.

Packet (JSON, about 400 bytes):
    {"v": 1, "seq": n, "t": laptop_time, "enable": bool, "arms": {joint: rad, ...}}
The robot (arms/arm_teleop.py on the Pi) echoes {"ack": seq, "t": t} back, so
the laptop can show the network round-trip time. A lost packet is never resent:
the next one, about 33 ms later, replaces it anyway.
"""
from __future__ import annotations

import json
import socket
import threading
import time


class RobotLink:
    def __init__(self, host, port=9874, dry=False):
        self.dry = dry
        self.seq = 0
        self.rtt_ms = None
        self.acks = 0
        self.last_ack = 0.0          # monotonic time of the last reply
        self.addr = None
        self.error = None
        if dry:
            return
        try:
            self.addr = (socket.gethostbyname(host), port)
        except OSError as e:
            self.error = f"can't resolve {host!r}: {e}"
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent_at = {}
        threading.Thread(target=self._acks, daemon=True).start()

    def send(self, enable, arms):
        self.seq += 1
        if self.dry or self.addr is None:
            return
        t = time.time()
        msg = {"v": 1, "seq": self.seq, "t": t, "enable": bool(enable),
               "arms": {k: round(float(v), 4) for k, v in arms.items()}}
        self.sent_at[self.seq] = time.monotonic()
        if len(self.sent_at) > 200:
            for k in sorted(self.sent_at)[:100]:
                del self.sent_at[k]
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
            self.error = None
        except OSError as e:
            self.error = str(e)

    def _acks(self):
        s = self.sock
        s.settimeout(0.5)
        s.setblocking(True)
        while True:
            try:
                data, _ = s.recvfrom(4096)
                ack = json.loads(data).get("ack")
            except (socket.timeout, ValueError):
                continue
            except OSError:
                time.sleep(0.2)
                continue
            t0 = self.sent_at.get(ack)
            if t0 is not None:
                rtt = (time.monotonic() - t0) * 1000
                self.rtt_ms = rtt if self.rtt_ms is None else 0.8 * self.rtt_ms + 0.2 * rtt
                self.acks += 1
                self.last_ack = time.monotonic()
