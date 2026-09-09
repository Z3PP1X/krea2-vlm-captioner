"""Thread-safe domain rate limiter enforcing polite intervals between HTTP requests."""

import time
import threading
from urllib.parse import urlparse
from typing import Dict


class DomainRateLimiter:
    """Enforces minimum delays between consecutive requests to the same domain."""

    def __init__(self, default_delay: float = 1.0):
        self.default_delay = max(0.0, default_delay)
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str, custom_delay: float | None = None) -> None:
        """Blocks the calling thread until the rate limit delay for the domain has passed."""
        delay = self.default_delay if custom_delay is None else max(0.0, custom_delay)
        if delay <= 0:
            return

        domain = urlparse(url).netloc.lower()
        with self._lock:
            now = time.time()
            last_time = self._last_access.get(domain, 0.0)
            elapsed = now - last_time
            wait_time = delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_access[domain] = time.time()
