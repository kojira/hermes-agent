"""Tests for the shared dollar usage model (agent/billing_usage.py).

Behavior contracts, not snapshots: status classification, bar math, fail-open,
and the dollars-only / topup-split invariants the billing UX feedback requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from agent.billing_usage import (
    LOW_BALANCE_THRESHOLD_USD,
    UsageBar,
    UsageModel,
    usage_model_from_account,
)


# ── Lightweight stand-ins for NousPortalAccountInfo shape ────────────────────


@dataclass
class _Access:
    subscription_credits_remaining: Optional[float] = None
    purchased_credits_remaining: Optional[float] = None
    total_usable_credits: Optional[float] = None


@dataclass
class _Sub:
    plan: Optional[str] = None
    monthly_credits: Optional[float] = None
    current_period_end: Optional[str] = None


@dataclass
class _Account:
    logged_in: bool = True
    paid_service_access: Optional[bool] = None
    paid_service_access_info: Optional[_Access] = None
    subscription: Optional[_Sub] = None


def _acct(**over):
    return _Account(**over)


# ── Fail-open ────────────────────────────────────────────────────────────────


def test_none_account_is_unavailable():
    assert usage_model_from_account(None).available is False


def test_logged_out_is_unavailable():
    assert usage_model_from_account(_acct(logged_in=False)).available is False


def test_garbage_account_fails_open():
    class Boom:
        @property
        def logged_in(self):
            raise RuntimeError("kaboom")

    assert usage_model_from_account(Boom()).available is False


# ── Status classification ────────────────────────────────────────────────────


def test_no_plan_no_balance_is_free():
    m = usage_model_from_account(_acct(paid_service_access=None, subscription=None, paid_service_access_info=_Access()))
    assert m.available is True
    assert m.status == "free"


def test_paid_access_lost_is_depleted():
    m = usage_model_from_account(
        _acct(
            paid_service_access=False,
            subscription=_Sub(plan="Plus", monthly_credits=20.0),
            paid_service_access_info=_Access(subscription_credits_remaining=0.0, total_usable_credits=0.0),
        )
    )
    assert m.status == "depleted"


def test_healthy_when_above_threshold():
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=20.0, current_period_end="2026-07-01"),
            paid_service_access_info=_Access(subscription_credits_remaining=14.0, total_usable_credits=14.0),
        )
    )
    assert m.status == "healthy"
    assert m.plan_name == "Plus"
    assert m.renews_at == "2026-07-01"


def test_low_when_total_spendable_under_threshold():
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=20.0),
            paid_service_access_info=_Access(subscription_credits_remaining=3.4, total_usable_credits=3.4),
        )
    )
    assert m.status == "low"
    assert m.is_low is True
    # Invariant: the threshold boundary is exclusive — exactly $5 is NOT low.
    assert LOW_BALANCE_THRESHOLD_USD == 5.0


def test_exactly_threshold_is_healthy():
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=20.0),
            paid_service_access_info=_Access(subscription_credits_remaining=5.0, total_usable_credits=5.0),
        )
    )
    assert m.status == "healthy"


def test_topup_only_no_subscription_is_not_free():
    # Purchased balance but no plan -> still a usable (healthy) account, not free.
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=None,
            paid_service_access_info=_Access(purchased_credits_remaining=30.0, total_usable_credits=30.0),
        )
    )
    assert m.status == "healthy"
    assert m.has_topup is True


# ── Bar math ─────────────────────────────────────────────────────────────────


def test_plan_bar_spent_and_pct():
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=20.0),
            paid_service_access_info=_Access(subscription_credits_remaining=14.0, total_usable_credits=14.0),
        )
    )
    bar = m.plan_bar
    assert bar is not None
    assert bar.kind == "plan"
    assert bar.remaining_usd == 14.0
    assert bar.total_usd == 20.0
    assert bar.spent_usd == pytest.approx(6.0)
    assert bar.pct_used == 30  # 6/20


def test_plan_bar_clamps_over_cap_remaining():
    # Rollover/debt: remaining > cap should clamp the bar's remaining to the cap
    # and read as zero spent, not a negative.
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=20.0),
            paid_service_access_info=_Access(subscription_credits_remaining=25.0, total_usable_credits=25.0),
        )
    )
    bar = m.plan_bar
    assert bar is not None
    assert bar.remaining_usd == 20.0
    assert bar.spent_usd == 0.0


def test_topup_bar_full_no_denominator():
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=20.0),
            paid_service_access_info=_Access(
                subscription_credits_remaining=14.0, purchased_credits_remaining=12.0, total_usable_credits=26.0
            ),
        )
    )
    tb = m.topup_bar
    assert tb is not None
    assert tb.kind == "topup"
    assert tb.remaining_usd == 12.0
    assert tb.fill_fraction == 1.0  # full bar = balance
    assert tb.pct_used is None  # no monthly denominator
    assert m.total_spendable_usd == 26.0


def test_no_plan_bar_without_monthly_denominator():
    # Top-up only, no monthly cap -> no plan bar, but a top-up bar.
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=None,
            paid_service_access_info=_Access(purchased_credits_remaining=8.0, total_usable_credits=8.0),
        )
    )
    assert m.plan_bar is None
    assert m.topup_bar is not None


def test_non_finite_values_are_ignored():
    m = usage_model_from_account(
        _acct(
            paid_service_access=True,
            subscription=_Sub(plan="Plus", monthly_credits=float("nan")),
            paid_service_access_info=_Access(subscription_credits_remaining=float("inf")),
        )
    )
    # NaN cap and Inf remaining must not produce a bar or a bogus total.
    assert m.plan_bar is None


# ── UsageBar property edge cases ─────────────────────────────────────────────


def test_usage_bar_fill_fraction_clamped():
    assert UsageBar(kind="plan", remaining_usd=30.0, total_usd=20.0).fill_fraction == 1.0
    assert UsageBar(kind="plan", remaining_usd=-5.0, total_usd=20.0).fill_fraction == 0.0
    assert UsageBar(kind="plan", remaining_usd=0.0, total_usd=0.0).fill_fraction == 0.0


def test_topup_bar_pct_used_is_none():
    assert UsageBar(kind="topup", remaining_usd=12.0, total_usd=12.0).pct_used is None
