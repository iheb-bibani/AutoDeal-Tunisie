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
