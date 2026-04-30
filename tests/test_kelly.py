"""Unit tests for Kelly Criterion sizing."""
import pytest

from src.betting.kelly import edge, kelly_stake


def test_kelly_returns_zero_on_no_edge():
    # Fair odds for 50% prob = 2.0. At odds=2.0, edge=0, no bet.
    assert kelly_stake(bankroll=1000, odds=2.0, probability=0.5) == 0.0


def test_kelly_returns_zero_on_negative_edge():
    # Bookmaker offers odds worse than fair: edge < 0 -> no bet
    assert kelly_stake(bankroll=1000, odds=1.5, probability=0.5) == 0.0


def test_kelly_full_stake_calculation():
    # b=1.5, p=0.6, q=0.4, full Kelly f* = (1.5*0.6 - 0.4)/1.5 = 0.333
    # 1/4 Kelly = 0.083, capped at 0.05 max.
    stake = kelly_stake(bankroll=1000, odds=2.5, probability=0.6, fraction=1.0, max_stake_pct=1.0)
    assert stake == pytest.approx(333.33, rel=0.01)


def test_quarter_kelly_caps_at_5pct():
    # Same setup but 1/4 Kelly with default 5% cap
    stake = kelly_stake(bankroll=10_000, odds=2.5, probability=0.6, fraction=0.25)
    # Full Kelly = 0.333; ¼ Kelly = 0.083; capped at 0.05 = $500
    assert stake == pytest.approx(500.0, rel=0.01)


def test_edge_function():
    assert edge(odds=2.0, probability=0.5) == pytest.approx(0.0)
    assert edge(odds=2.5, probability=0.5) == pytest.approx(0.25)
    assert edge(odds=1.5, probability=0.5) == pytest.approx(-0.25)


def test_kelly_zero_on_invalid_inputs():
    assert kelly_stake(bankroll=1000, odds=1.0, probability=0.5) == 0.0  # no payout
    assert kelly_stake(bankroll=1000, odds=2.0, probability=0.0) == 0.0  # impossible event
    assert kelly_stake(bankroll=1000, odds=2.0, probability=1.0) == 0.0  # certain event
