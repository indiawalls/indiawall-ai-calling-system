"""
Gemini API Key Rotator for IndiaWalls Voice Calling System.
Handles multi-account round-robin key rotation and automatic failover on 429 rate limit errors.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hindi-ai-calling.gemini-rotator")


class GeminiKeyRotator:
    """Manages rotation across multiple Google AI Studio Gemini API keys.
    Prevents single-key 15 RPM / 1,500 RPD rate limits by rotating across accounts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._index: int = 0
        self._cooldowns: Dict[str, float] = {}  # key -> cooldown end timestamp
        self._stats: Dict[str, Dict[str, int]] = {}  # key -> {requests, errors, 429s}
        self.reload_keys()

    def reload_keys(self):
        """Load keys from GEMINI_API_KEYS (comma separated) or GEMINI_API_KEY."""
        with self._lock:
            raw_multi = os.environ.get("GEMINI_API_KEYS", "").strip()
            raw_single = os.environ.get("GEMINI_API_KEY", "").strip()

            keys = []
            if raw_multi:
                keys = [k.strip() for k in raw_multi.split(",") if k.strip() and not k.strip().startswith("your-")]
            elif raw_single and not raw_single.startswith("your-"):
                keys = [raw_single]

            self._keys = keys
            self._index = 0
            for k in self._keys:
                if k not in self._stats:
                    self._stats[k] = {"requests": 0, "errors": 0, "rate_limits": 0}

            logger.info(f"GeminiKeyRotator initialized with {len(self._keys)} active key(s).")

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    def get_current_key(self) -> str:
        """Get the active key, rotating past any keys currently on cooldown."""
        with self._lock:
            if not self._keys:
                return os.environ.get("GEMINI_API_KEY", "")

            now = time.time()
            # Try to find a key not on cooldown
            for _ in range(len(self._keys)):
                key = self._keys[self._index]
                cooldown_until = self._cooldowns.get(key, 0.0)
                if now >= cooldown_until:
                    return key
                # Advance to next key if current is cooling down
                self._index = (self._index + 1) % len(self._keys)

            # If all are on cooldown, pick the one closest to expiry
            return self._keys[self._index]

    def get_next_key(self) -> str:
        """Advance round-robin to the next key and return it."""
        with self._lock:
            if not self._keys:
                return os.environ.get("GEMINI_API_KEY", "")

            self._index = (self._index + 1) % len(self._keys)
            key = self._keys[self._index]
            if key in self._stats:
                self._stats[key]["requests"] += 1
            return key

    def record_usage(self, key: str):
        """Record a successful request made with this key."""
        with self._lock:
            if key in self._stats:
                self._stats[key]["requests"] += 1

    def record_429(self, key: str, cooldown_secs: float = 60.0):
        """Put key on cooldown after receiving a 429 ResourceExhausted error
        and immediately switch active pointer to the next key."""
        with self._lock:
            now = time.time()
            self._cooldowns[key] = now + cooldown_secs
            if key in self._stats:
                self._stats[key]["rate_limits"] += 1
            logger.warning(
                f"Gemini key ending ...{key[-6:] if len(key) >= 6 else key} hit 429 rate limit! "
                f"Cooldown set for {cooldown_secs}s. Rotating to next key."
            )
            # Advance to next available key
            self._index = (self._index + 1) % len(self._keys)

    def get_status(self) -> Dict[str, Any]:
        """Return human-readable diagnostic status of all keys for dashboard."""
        with self._lock:
            now = time.time()
            key_details = []
            for i, k in enumerate(self._keys):
                cooldown_end = self._cooldowns.get(k, 0.0)
                remaining = max(0, int(cooldown_end - now))
                stats = self._stats.get(k, {"requests": 0, "rate_limits": 0})
                masked = f"...{k[-6:]}" if len(k) >= 6 else "***"
                key_details.append({
                    "id": i + 1,
                    "masked": masked,
                    "is_current": (i == self._index),
                    "on_cooldown": remaining > 0,
                    "cooldown_remaining_sec": remaining,
                    "requests": stats["requests"],
                    "rate_limits": stats["rate_limits"],
                })

            active_count = sum(1 for k in key_details if not k["on_cooldown"])
            return {
                "total_keys": len(self._keys),
                "active_keys": active_count,
                "cooldown_keys": len(self._keys) - active_count,
                "current_index": self._index + 1 if self._keys else 0,
                "keys": key_details,
            }


# Singleton instance shared across the application
gemini_rotator = GeminiKeyRotator()
