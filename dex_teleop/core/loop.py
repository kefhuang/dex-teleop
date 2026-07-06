from __future__ import annotations

import time


class RateLimiter:
    """Simple monotonic control-loop rate limiter."""

    def __init__(self, rate_hz: float) -> None:
        rate = float(rate_hz)
        if rate <= 0.0:
            raise ValueError("rate_hz must be positive")
        self.period = 1.0 / rate
        self._next_time = time.monotonic()

    def sleep(self) -> None:
        self._next_time += self.period
        remaining = self._next_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            self._next_time = time.monotonic()
