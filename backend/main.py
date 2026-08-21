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

import os
from functools import lru_cache
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.repository import MarketRepository
from backend.schemas import (
    ComparableMarketResponse,
    ComparableRequest,
    HealthResponse,
    MarketSummaryResponse,
)
from core.market_valuation import market_valuation

API_VERSION = "2.0.0"


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
    brand: str | None = None,
    model: str | None = None,
    max_price: Annotated[float | None, Query(gt=0)] = None,
    min_year: Annotated[int | None, Query(ge=1980, le=2100)] = None,
    max_mileage_km: Annotated[float | None, Query(ge=0)] = None,
    min_opportunity_pct: Annotated[float | None, Query(ge=-100, le=100)] = None,
    sort: Annotated[
        str,
        Query(pattern="^(opportunity|price_asc|price_desc|newest)$"),
    ] = "opportunity",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    repo: MarketRepository = Depends(get_repository),
) -> dict:
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

    if sort == "price_asc" and "Prix" in d.columns:
        d = d.sort_values("Prix", ascending=True)
    elif sort == "price_desc" and "Prix" in d.columns:
        d = d.sort_values("Prix", ascending=False)
    elif sort == "newest" and "Annonce-Detectee" in d.columns:
        d = d.sort_values("Annonce-Detectee", ascending=False)
    elif "Score_Opportunite" in d.columns:
        d = d.sort_values("Score_Opportunite", ascending=False)

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
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": _records(d, columns),
    }


@app.get("/api/v1/deals", tags=["deals"])
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
        d = d.sort_values("Score_Opportunite", ascending=False)
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
