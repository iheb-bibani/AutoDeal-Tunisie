"""Read-only data access for the FastAPI backend.

The API deliberately does not import Streamlit. Keeping data access framework-
agnostic lets the same service run under uvicorn, tests, a container, or a future
worker process.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import pandas as pd

from config import PROCESSED_FILES


@dataclass
class _CachedFrame:
    mtime_ns: int
    frame: pd.DataFrame


class MarketRepository:
    """Small mtime-aware CSV repository.

    GitHub Actions rewrites the CSVs in place. The repository reloads only when
    the file modification time changes, so API requests stay cheap without
    serving stale data forever.
    """

    def __init__(
        self,
        scored_path: str | Path | None = None,
        deals_path: str | Path | None = None,
    ) -> None:
        self.scored_path = Path(scored_path or PROCESSED_FILES["scored"])
        self.deals_path = Path(deals_path or PROCESSED_FILES["deals"])
        self._cache: dict[Path, _CachedFrame] = {}
        self._lock = RLock()

    def _read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        stat = path.stat()
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached.mtime_ns == stat.st_mtime_ns:
                return cached.frame.copy()
            frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
            self._cache[path] = _CachedFrame(stat.st_mtime_ns, frame)
            return frame.copy()

    def scored(self) -> pd.DataFrame:
        return self._read(self.scored_path)

    def deals(self) -> pd.DataFrame:
        return self._read(self.deals_path)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
