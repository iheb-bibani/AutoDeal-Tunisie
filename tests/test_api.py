import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app, get_repository


class FakeRepository:
    def __init__(self):
        rows = []
        prices = [45000, 47000, 49000, 50000, 52000, 54000, 56000]
        for i, price in enumerate(prices):
            rows.append({
                "Lien": f"https://example.test/208-{i}",
                "Source": "test-source",
                "Titre": f"Peugeot 208 {i}",
                "Marque": "Peugeot",
                "Modèle": "208",
                "Année": 2021 + (i % 2),
                "Age_Vehicule": 5 - (i % 2),
                "Kilométrage": 60000 + i * 2500,
                "Energie": "Essence",
                "Boite_Vitesse": "Manuelle",
                "Prix": price,
                "Prix_Theorique": 51000,
                "Score_Opportunite": (51000 - price) / 51000,
                "Nb_Comparables": 6,
                "Fiabilite_Estimation": "Moyenne",
                "Localisation": "Tunis",
                "Annonce-Detectee": "2026-08-21",
            })
        rows.append({
            "Lien": "https://example.test/golf",
            "Source": "another-source",
            "Titre": "Volkswagen Golf",
            "Marque": "Volkswagen",
            "Modèle": "Golf",
            "Année": 2020,
            "Age_Vehicule": 6,
            "Kilométrage": 80000,
            "Energie": "Diesel",
            "Boite_Vitesse": "Automatique",
            "Prix": 80000,
            "Prix_Theorique": 85000,
            "Score_Opportunite": 0.10,
            "Nb_Comparables": 1,
            "Fiabilite_Estimation": "Faible",
            "Localisation": "Sfax",
            "Annonce-Detectee": "2026-08-20",
        })
        self._scored = pd.DataFrame(rows)
        self._deals = self._scored[self._scored["Score_Opportunite"] > 0.05].copy()

    def scored(self):
        return self._scored.copy()

    def deals(self):
        return self._deals.copy()


def client():
    repo = FakeRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    with client() as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_market_summary():
    with client() as c:
        response = c.get("/api/v1/market/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["listings"] == 8
    assert body["sources"] == 2
    assert body["brands"] == 2
    assert body["models"] == 2


def test_listing_filters_and_sorting():
    with client() as c:
        response = c.get(
            "/api/v1/listings",
            params={"brand": "Peugeot", "max_price": 50000, "sort": "price_asc"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert [item["Prix"] for item in body["items"]] == [45000, 47000, 49000, 50000]
    assert {item["Marque"] for item in body["items"]} == {"Peugeot"}


def test_comparable_market_valuation():
    payload = {
        "brand": "Peugeot",
        "model": "208",
        "year": 2021,
        "mileage_km": 65000,
        "fuel": "Essence",
        "gearbox": "Manuelle",
        "min_comparables": 5,
    }
    with client() as c:
        response = c.post("/api/v1/valuation/comparables", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["n_comparables"] >= 5
    assert body["median_price_tnd"] is not None
    assert body["q25_tnd"] <= body["median_price_tnd"] <= body["q75_tnd"]
    assert all(row["Modèle"] == "208" for row in body["comparables"])
