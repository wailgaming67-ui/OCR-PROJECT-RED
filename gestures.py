from __future__ import annotations
import time
from collections import deque
from typing import Deque, Optional, Tuple

class WaveGestureDetector:

    def __init__(self, *, window_s: float=1.4, min_samples: int=18, min_reversals: int=4, min_span: float=0.07, cooldown_s: float=2.2) -> None:
        self._window_s = window_s
        self._min_samples = min_samples
        self._min_reversals = min_reversals
        self._min_span = min_span
        self._cooldown_s = cooldown_s
        self._buf: Deque[Tuple[float, float]] = deque()
        self._last_fire: float = 0.0

    def reset(self) -> None:
        self._buf.clear()

    def update(self, t: float, wrist_x_norm: Optional[float]) -> bool:
        if wrist_x_norm is None:
            self._buf.clear()
            return False
        if t - self._last_fire < self._cooldown_s:
            self._prune(t)
            return False
        self._buf.append((t, float(wrist_x_norm)))
        self._prune(t)
        if len(self._buf) < self._min_samples:
            return False
        xs = [p[1] for p in self._buf]
        span = max(xs) - min(xs)
        if span < self._min_span:
            return False
        n = len(xs)
        win = min(7, n // 3 | 1)
        if win < 3:
            centered = xs
        else:
            centered = []
            half = win // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                centered.append(xs[i] - sum(xs[lo:hi]) / (hi - lo))
        diffs = [centered[i + 1] - centered[i] for i in range(len(centered) - 1)]
        reversals = 0
        for i in range(len(diffs) - 1):
            if diffs[i] == 0:
                continue
            if diffs[i] * diffs[i + 1] < 0:
                reversals += 1
        if reversals >= self._min_reversals:
            self._last_fire = t
            self._buf.clear()
            return True
        return False

    def _prune(self, t: float) -> None:
        while self._buf and t - self._buf[0][0] > self._window_s:
            self._buf.popleft()

    @property
    def progress_hint(self) -> str:
        return f'buf={len(self._buf)}'
