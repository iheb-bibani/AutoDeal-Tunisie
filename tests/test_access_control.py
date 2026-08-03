from datetime import datetime, timedelta, timezone

from core.access_control import can_open_page, effective_plan, has_professional_access


def _future(days=30):
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def test_user_cannot_open_pro_pages():
    a = {"role": "user", "plan": "free", "subscription_status": "active"}
    assert can_open_page("🏠 Accueil", a)
    assert not can_open_page("🤝 Samsar", a)
    assert not can_open_page("🏢 Concessionnaire", a)


def test_samsar_pro_gets_samsar_not_dealer():
    # Un abonnement payant n'accorde des droits que pendant une période
    # effectivement payée et non expirée.
    a = {
        "role": "samsar",
        "plan": "pro",
        "subscription_status": "active",
        "current_period_end": _future(),
    }
    assert has_professional_access(a, "samsar")
    assert can_open_page("🤝 Samsar", a)
    assert not can_open_page("🏢 Concessionnaire", a)


def test_dealer_business_gets_dealer_not_samsar():
    a = {
        "role": "dealer",
        "plan": "business",
        "subscription_status": "active",
        "current_period_end": _future(),
    }
    assert has_professional_access(a, "dealer")
    assert can_open_page("🏢 Concessionnaire", a)
    assert not can_open_page("🤝 Samsar", a)


def test_inactive_paid_plan_falls_back_to_free():
    a = {
        "role": "samsar",
        "plan": "pro",
        "subscription_status": "cancelled",
        "current_period_end": _future(),
    }
    assert effective_plan(a) == "free"
    assert not can_open_page("🤝 Samsar", a)


def test_expired_paid_period_blocks_access_even_if_status_active():
    a = {
        "role": "samsar",
        "plan": "pro",
        "subscription_status": "active",
        "current_period_end": (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat(),
    }
    assert effective_plan(a) == "free"
    assert not has_professional_access(a, "samsar")
    assert not can_open_page("🤝 Samsar", a)


def test_admin_gets_everything():
    a = {"role": "admin", "plan": "free", "subscription_status": "past_due"}
    assert can_open_page("🤝 Samsar", a)
    assert can_open_page("🏢 Concessionnaire", a)
    assert can_open_page("🛠️ Admin", a)
