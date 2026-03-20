"""Base data collector with caching, rate limiting, and retry logic."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import urllib.request
import urllib.error
import urllib.parse

logger = logging.getLogger(__name__)

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / ".cache"


class AbstractCollector(ABC):
    """Base class for all data collectors."""

    def __init__(
        self,
        cache_dir: Optional[pathlib.Path] = None,
        rate_limit_per_second: float = 1.0,
        cache_ttl_hours: int = 24,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / rate_limit_per_second
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, url: str, params: dict) -> str:
        raw = url + json.dumps(params, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < self.cache_ttl_seconds:
                with open(path) as f:
                    return json.load(f)
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f)

    def fetch_json(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> Any:
        """Fetch JSON from URL with caching, rate limiting, and retry."""
        params = params or {}
        cache_key = self._cache_key(url, params)

        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {url}")
                return cached

        full_url = url
        if params:
            full_url = f"{url}?{urllib.parse.urlencode(params)}"

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                req = urllib.request.Request(full_url)
                req.add_header("User-Agent", "WestAfricaFTZ-Research/1.0")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    if use_cache:
                        self._set_cache(cache_key, data)
                    return data
            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limited
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif e.code >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"Server error {e.code}, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP error {e.code}: {url}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Error fetching {url}: {e}, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {url}")
                    raise

        return None

    @abstractmethod
    def collect(self, **kwargs) -> list[dict]:
        """Collect data and return as list of records."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this data source."""
        ...
