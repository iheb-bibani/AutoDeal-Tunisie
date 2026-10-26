import pandas as pd

from utils.send_personalized_alerts import _match


def row(**overrides):
    base = {
        "Marque": "Peugeot", "Modèle": "208", "Prix": 45000,
        "Kilométrage": 80000, "Année": 2021, "Score_Opportunite": 0.30,
    }
    base.update(overrides)
    return pd.Series(base)


def alert(**overrides):
    base = {
        "brand": "Peugeot", "model": "208", "budget_max": 50000,
        "max_km": 100000, "min_year": 2019, "min_gap_pct": 25,
    }
    base.update(overrides)
    return base


def test_match_all_criteria():
    assert _match(row(), alert()) is True


def test_match_rejects_budget_or_gap():
    assert _match(row(Prix=55000), alert()) is False
    assert _match(row(Score_Opportunite=0.20), alert()) is False


def test_match_accepts_optional_brand_model():
    assert _match(row(), alert(brand=None, model=None)) is True


def test_email_unique_obligatoire():
    from utils.send_personalized_alerts import valid_single_email

    assert valid_single_email("a@x.com") == "a@x.com"
    assert valid_single_email("Jean <a@x.com>") == "a@x.com"
    assert valid_single_email("a@x.com, b@y.com") is None
    assert valid_single_email("a@x.com\r\nBcc: b@y.com") is None
    assert valid_single_email("pas-un-email") is None
    assert valid_single_email(None) is None


def test_token_masque_dans_les_logs():
    from utils.send_personalized_alerts import _redact

    msg = "HTTPSConnectionPool: url: /bot123456:AAH-abc_def/sendMessage"
    assert "AAH" not in _redact(msg)


def test_message_telegram_deal_echappe_le_html():
    from utils.send_telegram import construire_message

    r = pd.Series({"Titre": "Golf_7 <GTI> *top*", "Marque": "Volkswagen", "Localisation": "Tunis"})
    msg = construire_message(r, 50000, 5000, 90000, "https://x.tn/a?b=1&c=2")
    assert "&lt;GTI&gt;" in msg
    assert 'href="https://x.tn/a?b=1&amp;c=2"' in msg
