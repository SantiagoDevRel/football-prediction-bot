"""Unit tests for the auto-resolver outcome rules."""
from src.tracking.auto_resolver import compute_outcome


def test_1x2_outcomes():
    # Home wins 2-1
    assert compute_outcome("1x2", "home", 2, 1) is True
    assert compute_outcome("1x2", "draw", 2, 1) is False
    assert compute_outcome("1x2", "away", 2, 1) is False

    # Draw 1-1
    assert compute_outcome("1x2", "home", 1, 1) is False
    assert compute_outcome("1x2", "draw", 1, 1) is True
    assert compute_outcome("1x2", "away", 1, 1) is False

    # Away wins 0-2
    assert compute_outcome("1x2", "home", 0, 2) is False
    assert compute_outcome("1x2", "draw", 0, 2) is False
    assert compute_outcome("1x2", "away", 0, 2) is True


def test_btts_outcomes():
    # Both scored
    assert compute_outcome("btts", "yes", 2, 1) is True
    assert compute_outcome("btts", "no", 2, 1) is False
    # 0-0
    assert compute_outcome("btts", "yes", 0, 0) is False
    assert compute_outcome("btts", "no", 0, 0) is True
    # 3-0
    assert compute_outcome("btts", "yes", 3, 0) is False
    assert compute_outcome("btts", "no", 3, 0) is True


def test_over_under_outcomes():
    # 1-1 (total 2): over 1.5 wins, under 2.5 wins
    assert compute_outcome("ou_1.5", "over", 1, 1) is True
    assert compute_outcome("ou_1.5", "under", 1, 1) is False
    assert compute_outcome("ou_2.5", "over", 1, 1) is False
    assert compute_outcome("ou_2.5", "under", 1, 1) is True

    # 2-2 (total 4)
    assert compute_outcome("ou_2.5", "over", 2, 2) is True
    assert compute_outcome("ou_3.5", "over", 2, 2) is True   # 4 > 3.5
    assert compute_outcome("ou_3.5", "under", 2, 2) is False
    # 1-2 (total 3)
    assert compute_outcome("ou_2.5", "over", 1, 2) is True
    assert compute_outcome("ou_3.5", "under", 1, 2) is True   # 3 < 3.5


def test_handicap_outcomes():
    # Home wins 3-0: home -1.5 wins (margin 3 >= 2)
    assert compute_outcome("ah_-1.5", "home", 3, 0) is True
    assert compute_outcome("ah_-1.5", "away", 3, 0) is False
    # Home wins 1-0: home -1.5 LOSES (margin 1 < 2), away +1.5 wins
    assert compute_outcome("ah_-1.5", "home", 1, 0) is False
    assert compute_outcome("ah_-1.5", "away", 1, 0) is True


def test_unknown_market_returns_none():
    assert compute_outcome("nonsense", "x", 1, 1) is None
