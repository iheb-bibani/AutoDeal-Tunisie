"""FastAPI application for AutoDeal Tunisie.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Interactive documentation:
    http://127.0.0.1:8000/docs

This API is intentionally read-only. Authentication, favorites and paid-plan
mutations remain behind Supabase/RLS until dedicated backend endpoints are
introduced with proper authorization.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.repository import MarketRepository
from backend.schemas import (
    ComparableMarketResponse,
    ComparableRequest,
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    MarketSummaryResponse,
)
from core.inference import features_depuis_saisie, preparer_lignes
from core.intervalles import intervalle
from core.market_valuation import market_valuation

API_VERSION = "2.1.0"
RATE_LIMIT_PER_MIN = int(os.getenv("AUTODEAL_RATE_LIMIT_PER_MIN", "120"))


def cle_api_valide(x_api_key: str | None) -> bool:
    """True si aucune clé n'est configurée, ou si la clé fournie est la bonne."""
    attendue = os.getenv("AUTODEAL_API_KEY")
    if not attendue:
        return True
    return bool(x_api_key) and hmac.compare_digest(x_api_key.encode(), attendue.encode())


# Colonnes qui constituent le radar de deals : réservées aux détenteurs de clé.
COLONNES_RADAR = ("Prix_Theorique", "Score_Opportunite", "Nb_Comparables", "Fiabilite_Estimation",
                  "Prix_Bas_80", "Prix_Haut_80")


def exiger_cle_api(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Protège les données à valeur commerciale (radar de deals, tri par
    opportunité) quand ``AUTODEAL_API_KEY`` est défini. Sans variable, l'API
    reste ouverte (développement local, compatibilité)."""
    if not cle_api_valide(x_api_key):
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide (en-tête X-API-Key).")


class _LimiteurDebit:
    """Fenêtre glissante d'une minute par adresse IP (en mémoire, par process)."""

    def __init__(self, par_minute: int):
        self.par_minute = par_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def autoriser(self, cle: str) -> bool:
        if self.par_minute <= 0:
            return True
        maintenant = time.monotonic()
        with self._lock:
            if len(self._hits) > self.MAX_CLES:
                self._purger(maintenant)
            hits = self._hits[cle]
            while hits and maintenant - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.par_minute:
                return False
            hits.append(maintenant)
            return True

    MAX_CLES = 10_000

    def _purger(self, maintenant: float) -> None:
        """Supprime les IP inactives depuis plus d'une minute (mémoire bornée)."""
        for cle in [k for k, h in self._hits.items() if not h or maintenant - h[-1] > 60]:
            del self._hits[cle]


limiteur = _LimiteurDebit(RATE_LIMIT_PER_MIN)


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "AUTODEAL_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_repository() -> MarketRepository:
    return MarketRepository()


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    view = df[columns].copy() if columns else df.copy()
    rows: list[dict] = []
    for record in view.to_dict(orient="records"):
        rows.append({str(k): _json_value(v) for k, v in record.items()})
    return rows


app = FastAPI(
    title="AutoDeal Tunisie API",
    description=(
        "Read-only market intelligence API for Tunisian used-car listings, "
        "deals and independent comparable-market valuation."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
@app.middleware("http")
async def limiter_debit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client = request.client.host if request.client else "inconnu"
        if not limiteur.autoriser(client):
            return JSONResponse(status_code=429, content={"detail": "Trop de requêtes, réessayez dans une minute."},
                                headers={"Retry-After": "60"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    # API publique sans cookie ni session : pas de credentials cross-origin.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "AutoDeal Tunisie API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="autodeal-api", version=API_VERSION)


@app.get(
    "/api/v1/market/summary",
    response_model=MarketSummaryResponse,
    tags=["market"],
)
def market_summary(
    repo: MarketRepository = Depends(get_repository),
) -> MarketSummaryResponse:
    df = repo.scored()
    if df.empty:
        raise HTTPException(status_code=503, detail="Scored market data is unavailable")

    price = pd.to_numeric(df.get("Prix"), errors="coerce")
    detection = df.get("Annonce-Detectee")
    latest = pd.to_datetime(detection, errors="coerce").max() if detection is not None else pd.NaT
    models = 0
    if {"Marque", "Modèle"}.issubset(df.columns):
        models = int(df[["Marque", "Modèle"]].dropna().drop_duplicates().shape[0])
    return MarketSummaryResponse(
        listings=int(len(df)),
        sources=int(df["Source"].nunique()) if "Source" in df.columns else 0,
        brands=int(df["Marque"].nunique()) if "Marque" in df.columns else 0,
        models=models,
        median_price_tnd=None if price.dropna().empty else float(price.median()),
        latest_detection=None if pd.isna(latest) else latest.isoformat(),
    )


@app.get("/api/v1/listings", tags=["listings"])
def listings(
    brand: Annotated[str | None, Query(max_length=64)] = None,
    model: Annotated[str | None, Query(max_length=64)] = None,
    max_price: Annotated[float | None, Query(gt=0)] = None,
    min_year: Annotated[int | None, Query(ge=1980, le=2100)] = None,
    max_mileage_km: Annotated[float | None, Query(ge=0)] = None,
    min_opportunity_pct: Annotated[float | None, Query(ge=-100, le=100)] = None,
    sort: Annotated[
        str | None,
        Query(pattern="^(opportunity|price_asc|price_desc|newest)$",
              description="Défaut : opportunity avec clé API valide, newest sinon."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    repo: MarketRepository = Depends(get_repository),
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    # Le classement par opportunité et les scores SONT le radar de deals.
    # Sans clé valide : tri par défaut "newest", scores retirés de la réponse
    # (sinon un tri client sur Score_Opportunite reconstruisait le radar).
    radar = cle_api_valide(x_api_key)
    if sort is None:
        sort = "opportunity" if radar else "newest"
    if not radar and (sort == "opportunity" or min_opportunity_pct is not None):
        raise HTTPException(status_code=401, detail="Clé API requise pour le tri par opportunité (en-tête X-API-Key).")
    df = repo.scored()
    if df.empty:
        return {"total": 0, "offset": offset, "limit": limit, "items": []}
    d = df.copy()
    if brand and "Marque" in d.columns:
        d = d[d["Marque"].astype(str).str.casefold() == brand.casefold()]
    if model and "Modèle" in d.columns:
        d = d[d["Modèle"].astype(str).str.casefold() == model.casefold()]
    if max_price is not None and "Prix" in d.columns:
        d = d[pd.to_numeric(d["Prix"], errors="coerce") <= max_price]
    if min_year is not None and "Année" in d.columns:
        d = d[pd.to_numeric(d["Année"], errors="coerce") >= min_year]
    if max_mileage_km is not None and "Kilométrage" in d.columns:
        d = d[pd.to_numeric(d["Kilométrage"], errors="coerce") <= max_mileage_km]
    if min_opportunity_pct is not None and "Score_Opportunite" in d.columns:
        d = d[
            pd.to_numeric(d["Score_Opportunite"], errors="coerce")
            >= min_opportunity_pct / 100.0
        ]

    # Tri sur des clés typées : une colonne relue en texte se trierait sinon
    # lexicographiquement ("9000" > "45000").
    if sort == "price_asc" and "Prix" in d.columns:
        d = d.sort_values("Prix", ascending=True, key=lambda s: pd.to_numeric(s, errors="coerce"))
    elif sort == "price_desc" and "Prix" in d.columns:
        d = d.sort_values("Prix", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    elif sort == "newest" and "Annonce-Detectee" in d.columns:
        d = d.sort_values(
            "Annonce-Detectee", ascending=False, key=lambda s: pd.to_datetime(s, errors="coerce", utc=True)
        )
    elif "Score_Opportunite" in d.columns:
        d = d.sort_values(
            "Score_Opportunite", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce")
        )

    total = int(len(d))
    d = d.iloc[offset : offset + limit]
    columns = [
        c
        for c in [
            "Lien",
            "Source",
            "Titre",
            "Marque",
            "Modèle",
            "Année",
            "Prix",
            "Kilométrage",
            "Energie",
            "Boite_Vitesse",
            "Localisation",
            "Prix_Theorique",
            "Score_Opportunite",
            "Nb_Comparables",
            "Fiabilite_Estimation",
            "Annonce-Detectee",
        ]
        if c in d.columns
    ]
    if not radar:
        columns = [c for c in columns if c not in COLONNES_RADAR]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": _records(d, columns),
    }


@app.get("/api/v1/deals", tags=["deals"], dependencies=[Depends(exiger_cle_api)])
def deals(
    min_comparables: Annotated[int, Query(ge=0, le=1000)] = 5,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    repo: MarketRepository = Depends(get_repository),
) -> dict:
    d = repo.deals()
    if d.empty:
        return {"total": 0, "items": []}
    if "Nb_Comparables" in d.columns:
        d = d[
            pd.to_numeric(d["Nb_Comparables"], errors="coerce").fillna(0)
            >= min_comparables
        ]
    if "Score_Opportunite" in d.columns:
        d = d.sort_values("Score_Opportunite", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    total = int(len(d))
    columns = [
        c
        for c in [
            "Lien",
            "Source",
            "Titre",
            "Marque",
            "Modèle",
            "Année",
            "Prix",
            "Prix_Theorique",
            "Score_Opportunite",
            "Nb_Comparables",
            "Fiabilite_Estimation",
            "Localisation",
            "Annonce-Detectee",
        ]
        if c in d.columns
    ]
    return {"total": total, "items": _records(d.head(limit), columns)}


@app.post(
    "/api/v1/valuation/comparables",
    response_model=ComparableMarketResponse,
    tags=["valuation"],
)
def comparable_valuation(
    payload: ComparableRequest,
    repo: MarketRepository = Depends(get_repository),
) -> ComparableMarketResponse:
    df = repo.scored()
    if df.empty:
        raise HTTPException(status_code=503, detail="Scored market data is unavailable")

    target = {
        "Marque": payload.brand,
        "Modèle": payload.model,
        "Année": payload.year,
        "Kilométrage": payload.mileage_km,
        "Energie": payload.fuel,
        "Boite_Vitesse": payload.gearbox,
        "Lien": payload.listing_url,
    }
    market, comps = market_valuation(
        df,
        target,
        min_n=payload.min_comparables,
        max_n=payload.max_comparables,
    )
    columns = [
        c
        for c in [
            "Lien",
            "Source",
            "Marque",
            "Modèle",
            "Année",
            "Kilométrage",
            "Energie",
            "Boite_Vitesse",
            "Prix",
            "Localisation",
        ]
        if c in comps.columns
    ]
    return ComparableMarketResponse(
        n_comparables=market.n_comparables,
        median_price_tnd=market.median_price,
        p10_tnd=market.p10,
        q25_tnd=market.q25,
        q75_tnd=market.q75,
        p90_tnd=market.p90,
        relative_width=market.relative_width,
        homogeneity=market.homogeneity,
        confidence=market.confidence,
        selection_level=market.selection_level,
        comparables=_records(comps, columns),
    )


@app.post("/api/v1/valuation/estimate", response_model=EstimateResponse, tags=["valuation"])
def estimate(
    payload: EstimateRequest,
    repo: MarketRepository = Depends(get_repository),
) -> EstimateResponse:
    """Estimation ML d'un véhicule + fourchette calibrée à 80 % + contrôle par
    les comparables réels (indépendant du modèle)."""
    bundle = repo.model()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Price model is unavailable")
    saisie = features_depuis_saisie(
        payload.brand, payload.model, payload.year, payload.mileage_km,
        payload.fuel, payload.gearbox, payload.fiscal_power_cv, payload.engine_liters,
    )
    prix = float(np.expm1(bundle["pipeline"].predict(preparer_lignes(bundle, saisie))[0]))
    calib = bundle.get("intervalles")
    bas, haut = intervalle(prix, calib) if calib else (None, None)

    df = repo.scored()
    market = None
    if not df.empty:
        market, _ = market_valuation(df, {
            "Marque": payload.brand, "Modèle": payload.model, "Année": payload.year,
            "Age_Vehicule": saisie["Age_Vehicule"], "Kilométrage": payload.mileage_km,
            "Energie": payload.fuel, "Boite_Vitesse": payload.gearbox,
        }, min_n=5)
    return EstimateResponse(
        estimate_tnd=round(prix, -2),
        low_80_tnd=None if bas is None else round(bas, -2),
        high_80_tnd=None if haut is None else round(haut, -2),
        interval_level=None if not calib else calib.get("niveau"),
        model_name=str(bundle.get("modele_nom", "inconnu")),
        model_trained_at=bundle.get("date_entrainement"),
        model_mdape=bundle.get("mdape_oof"),
        market_comparables=0 if market is None else int(market.n_comparables),
        market_median_tnd=None if market is None else _json_value(market.median_price),
        market_q25_tnd=None if market is None else _json_value(market.q25),
        market_q75_tnd=None if market is None else _json_value(market.q75),
    )
