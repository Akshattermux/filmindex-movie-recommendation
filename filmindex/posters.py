"""Retrieve public poster URLs. The read token stays on the Streamlit server."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests


_LOG = logging.getLogger(__name__)
_POSTER_PATH = re.compile(r"/[A-Za-z0-9_-]+\.(?:jpg|png|webp)", re.IGNORECASE)
_CACHE_LIMIT = 6000
_SUCCESS_TTL = 24 * 60 * 60
_MISSING_TTL = 60 * 60
_FAILURE_TTL = 30


def _retry_after(value: str | None) -> float:
    if value:
        if value.isascii() and value.isdigit() and len(value) <= 10:
            return max(1, int(value))
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is not None:
                return max(1, (when - datetime.now(timezone.utc)).total_seconds())
        except (OverflowError, TypeError, ValueError):
            pass
    return _FAILURE_TTL


class _PosterCache:
    """Bound outgoing requests across sessions and suppress duplicate in-flight work."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: set[int] = set()
        self.entries: OrderedDict[int, tuple[float, str | None]] = OrderedDict()
        self.retry_at = 0.0
        self.rejected_token: tuple[bytes, float] | None = None

    def get(self, movie_id: int, token: str) -> str | None:
        if not token or movie_id <= 0:
            return None

        fingerprint = hashlib.sha256(token.encode()).digest()
        with self.lock:
            now = time.monotonic()
            entry = self.entries.get(movie_id)
            if entry and entry[0] > now:
                self.entries.move_to_end(movie_id)
                return entry[1]
            if now < self.retry_at or movie_id in self.active or len(self.active) >= 5:
                return None
            if self.rejected_token:
                rejected, expires = self.rejected_token
                if fingerprint == rejected and now < expires:
                    return None
            self.active.add(movie_id)

        try:
            return self._fetch(movie_id, token, fingerprint)
        finally:
            with self.lock:
                self.active.remove(movie_id)

    def _remember(self, movie_id: int, url: str | None, seconds: float) -> str | None:
        with self.lock:
            self.entries[movie_id] = (time.monotonic() + seconds, url)
            self.entries.move_to_end(movie_id)
            if len(self.entries) > _CACHE_LIMIT:
                self.entries.popitem(last=False)
        return url

    def _fetch(self, movie_id: int, token: str, fingerprint: bytes) -> str | None:
        try:
            with requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                params={"language": "en-US"},
                timeout=(1.5, 2.5),
                allow_redirects=False,
            ) as response:
                status = response.status_code
                if status in (401, 403):
                    with self.lock:
                        self.rejected_token = fingerprint, time.monotonic() + _FAILURE_TTL
                    _LOG.warning("TMDB rejected the poster credential (HTTP %d)", status)
                    return None
                if status in (429, 503):
                    until = time.monotonic() + _retry_after(response.headers.get("Retry-After"))
                    with self.lock:
                        self.retry_at = max(self.retry_at, until)
                    _LOG.warning("TMDB poster lookups paused after HTTP %d", status)
                    return None
                if status == 404:
                    return self._remember(movie_id, None, _MISSING_TTL)
                if status != 200:
                    _LOG.warning("TMDB poster lookup returned HTTP %d for %d", status, movie_id)
                    return self._remember(movie_id, None, _FAILURE_TTL)
                payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Exception messages can include request data; log only their type.
            _LOG.warning("TMDB poster lookup for %d failed (%s)", movie_id, type(exc).__name__)
            return self._remember(movie_id, None, _FAILURE_TTL)

        path = payload.get("poster_path") if isinstance(payload, dict) else None
        if not isinstance(path, str) or not _POSTER_PATH.fullmatch(path):
            return self._remember(movie_id, None, _MISSING_TTL)
        return self._remember(movie_id, f"https://image.tmdb.org/t/p/w500{path}", _SUCCESS_TTL)


_cache = _PosterCache()


def get_poster_url(movie_id: int, token: str) -> str | None:
    """Return a poster URL or None when artwork or upstream capacity is unavailable."""
    return _cache.get(movie_id, token.strip())
