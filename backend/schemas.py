"""Pydantic contracts for AutoDeal's public read-only API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ComparableRequest(BaseModel):
    brand: str = Field(min_length=1, examples=["Peugeot"])
    model: str = Field(min_length=1, examples=["208"])
    year: int | None = Field(default=None, ge=1980, le=2100)
    mileage_km: float | None = Field(default=None, ge=0, le=1_500_000)
    fuel: str | None = None
    gearbox: str | None = None
    listing_url: str | None = None
    min_comparables: int = Field(default=5, ge=3, le=30)
    max_comparables: int = Field(default=40, ge=5, le=100)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class MarketSummaryResponse(BaseModel):
    listings: int
    sources: int
    brands: int
    models: int
    median_price_tnd: float | None
    latest_detection: str | None


class ComparableMarketResponse(BaseModel):
    n_comparables: int
    median_price_tnd: float | None
    p10_tnd: float | None
    q25_tnd: float | None
    q75_tnd: float | None
    p90_tnd: float | None
    relative_width: float | None
    homogeneity: str
    confidence: str
    selection_level: str
    comparables: list[dict]
