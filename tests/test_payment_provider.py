import services.payment_provider as pp


def _patch_secrets(monkeypatch, values):
    monkeypatch.setattr(pp, "_secret", lambda name: values.get(name))


def test_checkout_disabled_without_confirmation_webhook(monkeypatch):
    _patch_secrets(monkeypatch, {
        "KONNECT_API_KEY": "key",
        "KONNECT_WALLET_ID": "wallet",
    })
    availability = pp.checkout_availability()
    assert availability.enabled is False
    assert "KONNECT_WEBHOOK_URL" in availability.reason


def test_checkout_enabled_with_provider_and_webhook(monkeypatch):
    _patch_secrets(monkeypatch, {
        "KONNECT_API_KEY": "key",
        "KONNECT_WALLET_ID": "wallet",
        "KONNECT_WEBHOOK_URL": "https://example.test/webhook",
    })
    availability = pp.checkout_availability()
    assert availability.enabled is True
    assert availability.provider == "Konnect"


def test_plan_reserve_aux_roles_concernes():
    assert pp.plan_allowed_for_role("pro", "samsar")
    assert not pp.plan_allowed_for_role("pro", "user")
    assert not pp.plan_allowed_for_role("business", "samsar")
    assert pp.plan_allowed_for_role("business_plus", "dealer")


def test_checkout_refuse_plan_inutile_pour_le_role(monkeypatch):
    import pytest

    _patch_secrets(monkeypatch, {
        "KONNECT_API_KEY": "key",
        "KONNECT_WALLET_ID": "wallet",
        "KONNECT_WEBHOOK_URL": "https://example.test/webhook",
    })
    with pytest.raises(pp.CheckoutError):
        pp.start_checkout(user_id="u", plan="pro", role="user")


def test_prix_webhook_identiques_au_python():
    """Le webhook Deno recontrôle le montant : ses prix doivent être ceux du Python."""
    import re
    from pathlib import Path

    ts = (Path(__file__).resolve().parents[1] / "supabase/functions/konnect-webhook/index.ts").read_text()
    for plan, cycles in pp.PLAN_PRICES_TND.items():
        m = re.search(rf"{plan}: \{{ monthly: (\d+), yearly: (\d+) \}}", ts)
        assert m, f"prix {plan} absent du webhook"
        assert (int(m.group(1)), int(m.group(2))) == (cycles["monthly"], cycles["yearly"])


def test_prix_payment_provider_identiques_au_catalogue():
    from core.access_control import PLAN_CATALOG

    for plan, cycles in pp.PLAN_PRICES_TND.items():
        assert PLAN_CATALOG[plan]["monthly_tnd"] == cycles["monthly"]
        assert PLAN_CATALOG[plan]["yearly_tnd"] == cycles["yearly"]
