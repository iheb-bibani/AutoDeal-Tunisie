import pytest
import requests

import utils.subscription_maintenance as sm

TOKEN = "123456789:AAH-secret_token"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload

    def json(self):
        return self._payload


def test_erreur_telegram_ne_contient_jamais_le_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", TOKEN)
    monkeypatch.setattr(
        sm.requests, "post",
        lambda url, **kw: _Resp(400, {"description": f"Bad Request: chat not found (bot{TOKEN})"}),
    )
    with pytest.raises(sm.DeliveryError) as exc:
        sm._send_telegram("42", "hello")
    assert "secret" not in str(exc.value)
    assert "telegram_http_400" in str(exc.value)


def test_erreur_reseau_telegram_ne_contient_pas_l_url(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", TOKEN)

    def boom(url, **kw):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    monkeypatch.setattr(sm.requests, "post", boom)
    with pytest.raises(sm.DeliveryError) as exc:
        sm._send_telegram("42", "hello")
    assert TOKEN not in str(exc.value)


class _Query:
    """Faux client PostgREST : enregistre les filtres et renvoie des données fixes."""

    def __init__(self, store, table):
        self.store, self.table, self.filters, self.op = store, table, [], "select"

    def select(self, *_):
        return self

    def update(self, values):
        self.op, self.values = "update", values
        return self

    def insert(self, values):
        self.op, self.values = "insert", values
        return self

    def __getattr__(self, name):  # eq, neq, lte, in_...
        def f(*args):
            self.filters.append((name, *args))
            return self
        return f

    def execute(self):
        self.store["calls"].append((self.table, self.op, self.filters, getattr(self, "values", None)))
        data = self.store["data"].get((self.table, self.op), [])
        return type("R", (), {"data": data})()


class _FakeSb:
    def __init__(self, data):
        self.store = {"calls": [], "data": data}

    def table(self, name):
        return _Query(self.store, name)


def test_expiration_conditionnelle_et_notification_si_ligne_modifiee():
    row = {"user_id": "u1", "plan": "pro", "status": "active", "current_period_end": "2020-01-01T00:00:00Z"}
    sb = _FakeSb({("subscriptions", "select"): [row], ("subscriptions", "update"): [row]})
    sm.expire_due_subscriptions(sb)
    updates = [c for c in sb.store["calls"] if c[0] == "subscriptions" and c[1] == "update"]
    # 1re mise à jour = expiration ; elle doit re-filtrer sur status + date.
    assert ("eq", "status", "active") in updates[0][2]
    assert any(f[0] == "lte" and f[1] == "current_period_end" for f in updates[0][2])
    assert any(c[0] == "subscription_notifications" for c in sb.store["calls"])


def test_pas_de_notification_si_renouvele_entre_temps():
    row = {"user_id": "u1", "plan": "pro", "status": "active", "current_period_end": "2020-01-01T00:00:00Z"}
    sb = _FakeSb({("subscriptions", "select"): [row], ("subscriptions", "update"): []})
    sm.expire_due_subscriptions(sb)
    assert not any(c[0] == "subscription_notifications" for c in sb.store["calls"])
