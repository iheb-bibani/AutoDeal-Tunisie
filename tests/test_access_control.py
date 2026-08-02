from datetime import datetime, timedelta, timezone

from core.access_control import can_open_page, effective_plan, has_professional_access


def _futur(jours: int = 30) -> str:
    """Date ISO dans le futur — un plan payant n'est actif que pendant sa
    période effectivement réglée (current_period_end > maintenant)."""
    return (datetime.now(timezone.utc) + timedelta(days=jours)).isoformat()


def test_user_cannot_open_pro_pages():
    a = {"role": "user", "plan": "free", "subscription_status": "active"}
    assert can_open_page("🏠 Accueil", a)
    assert not can_open_page("🤝 Samsar", a)
    assert not can_open_page("🏢 Concessionnaire", a)


def test_samsar_pro_gets_samsar_not_dealer():
    a = {
        "role": "samsar",
        "plan": "pro",
        "subscription_status": "active",
        "current_period_end": _futur(),
    }
    assert has_professional_access(a, "samsar")
    assert can_open_page("🤝 Samsar", a)
    assert not can_open_page("🏢 Concessionnaire", a)


def test_dealer_business_gets_dealer_not_samsar():
    a = {
        "role": "dealer",
        "plan": "business",
        "subscription_status": "active",
        "current_period_end": _futur(),
    }
    assert has_professional_access(a, "dealer")
    assert can_open_page("🏢 Concessionnaire", a)
    assert not can_open_page("🤝 Samsar", a)


def test_inactive_paid_plan_falls_back_to_free():
    a = {"role": "samsar", "plan": "pro", "subscription_status": "cancelled"}
    assert effective_plan(a) == "free"
    assert not can_open_page("🤝 Samsar", a)


def test_admin_gets_everything():
    a = {"role": "admin", "plan": "free", "subscription_status": "active"}
    assert can_open_page("🤝 Samsar", a)
    assert can_open_page("🏢 Concessionnaire", a)
    assert can_open_page("🛠️ Admin", a)
