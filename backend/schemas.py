"""Pydantic contracts for AutoDeal's public read-only API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ComparableRequest(BaseModel):
    brand: str = Field(min_length=1, max_length=64, examples=["Peugeot"])
    model: str = Field(min_length=1, max_length=64, examples=["208"])
    year: int | None = Field(default=None, ge=1980, le=2100)
    mileage_km: float | None = Field(default=None, ge=0, le=1_500_000)
    fuel: str | None = Field(default=None, max_length=32)
    gearbox: str | None = Field(default=None, max_length=32)
    listing_url: str | None = Field(default=None, max_length=512)
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


class EstimateRequest(BaseModel):
    brand: str = Field(min_length=1, max_length=64, examples=["Peugeot"])
    model: str = Field(min_length=1, max_length=64, examples=["208"])
    year: int = Field(ge=1980, le=2100, examples=[2019])
    mileage_km: float = Field(ge=0, le=1_500_000, examples=[85000])
    fuel: str | None = Field(default=None, max_length=32, examples=["Essence"])
    gearbox: str | None = Field(default=None, max_length=32, examples=["Manuelle"])
    fiscal_power_cv: float | None = Field(default=None, ge=1, le=30, examples=[5])
    engine_liters: float | None = Field(default=None, ge=0.6, le=8.0, examples=[1.2])


class EstimateResponse(BaseModel):
    estimate_tnd: float
    low_80_tnd: float | None
    high_80_tnd: float | None
    interval_level: float | None
    model_name: str
    model_trained_at: str | None
    model_mdape: float | None
    market_comparables: int
    market_median_tnd: float | None
    market_q25_tnd: float | None
    market_q75_tnd: float | None
