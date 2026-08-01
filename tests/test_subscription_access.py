from datetime import datetime, timedelta, timezone

from core.access_control import subscription_is_active, effective_plan


def iso(delta_days):
    return (datetime.now(timezone.utc) + timedelta(days=delta_days)).isoformat()


def test_paid_plan_requires_unexpired_period():
    access = {"role": "samsar", "plan": "pro", "subscription_status": "active", "current_period_end": iso(-1)}
    assert subscription_is_active(access) is False
    assert effective_plan(access) == "free"


def test_paid_plan_active_during_paid_period():
    access = {"role": "dealer", "plan": "business", "subscription_status": "active", "current_period_end": iso(10)}
    assert subscription_is_active(access) is True
    assert effective_plan(access) == "business"


def test_admin_not_blocked_by_billing():
    assert subscription_is_active({"role": "admin", "plan": "business_plus", "subscription_status": "past_due"}) is True
