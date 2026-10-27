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

from config import MODEL_PATH, PROCESSED_FILES


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
        model_path: str | Path | None = None,
    ) -> None:
        self.scored_path = Path(scored_path or PROCESSED_FILES["scored"])
        self.deals_path = Path(deals_path or PROCESSED_FILES["deals"])
        self.model_path = Path(model_path or MODEL_PATH)
        self._cache: dict[Path, _CachedFrame] = {}
        self._model: tuple[int, dict] | None = None
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

    def model(self) -> dict | None:
        """Bundle du modèle de prix, rechargé quand le fichier change.

        Le pickle est produit par le pipeline de ce dépôt (source de confiance) ;
        ne jamais pointer ``model_path`` vers un fichier fourni par un tiers.
        """
        if not self.model_path.exists():
            return None
        mtime = self.model_path.stat().st_mtime_ns
        with self._lock:
            if self._model and self._model[0] == mtime:
                return self._model[1]
            import joblib

            bundle = joblib.load(self.model_path)
            if not (isinstance(bundle, dict) and "pipeline" in bundle):
                return None
            self._model = (mtime, bundle)
            return bundle

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._model = None
