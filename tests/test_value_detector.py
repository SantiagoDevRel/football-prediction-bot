"""Unit tests for value detection."""
import pytest

from src.betting.value_detector import OddsLine, detect_value, _select_prob
from src.models.base import MatchProbabilities


def make_pred(p_home=0.5, p_draw=0.25, p_away=0.25, **kw) -> MatchProbabilities:
    return MatchProbabilities(
        p_home_win=p_home, p_draw=p_draw, p_away_win=p_away,
        p_over_2_5=kw.get("p_over_2_5", 0.5), p_under_2_5=kw.get("p_under_2_5", 0.5),
        p_btts_yes=kw.get("p_btts_yes", 0.5), p_btts_no=kw.get("p_btts_no", 0.5),
        expected_home_goals=1.5, expected_away_goals=1.0,
    )


def test_detects_value_above_min_edge():
    pred = make_pred(p_home=0.6)  # 60% home -> fair odds = 1.67
    odds = [OddsLine("1x2", "home", 2.0)]  # casa pays 2.0 -> edge = 1.20*1 = +20%
    bets = detect_value(
        match_id=1, home_team="A", away_team="B", league="X",
        prediction=pred, odds_lines=odds, bankroll=1000,
        min_edge=0.05, max_edge=0.30,
    )
    assert len(bets) == 1
    assert bets[0].selection == "home"
    assert bets[0].edge == pytest.approx(0.20, abs=0.01)


def test_filters_below_min_edge():
    pred = make_pred(p_home=0.55)  # 55% home, fair odds 1.82
    odds = [OddsLine("1x2", "home", 1.85)]  # edge ~1.7%, below 5%
    bets = detect_value(
        match_id=1, home_team="A", away_team="B", league="X",
        prediction=pred, odds_lines=odds, bankroll=1000,
    )
    assert bets == []


def test_filters_above_max_edge():
    # Suspiciously high edge — likely model error, not real value
    pred = make_pred(p_home=0.7)  # 70% home, fair odds 1.43
    odds = [OddsLine("1x2", "home", 5.0)]  # edge ~250% -> filtered
    bets = detect_value(
        match_id=1, home_team="A", away_team="B", league="X",
        prediction=pred, odds_lines=odds, bankroll=1000,
        max_edge=0.30,
    )
    assert bets == []


def test_selects_correct_market_probabilities():
    pred = make_pred(p_home=0.55, p_over_2_5=0.65, p_btts_yes=0.7)
    assert _select_prob(pred, "1x2", "home") == 0.55
    assert _select_prob(pred, "ou_2.5", "over") == 0.65
    assert _select_prob(pred, "btts", "yes") == 0.7
    assert _select_prob(pred, "unknown", "x") is None


def test_sorts_by_edge_descending():
    pred = make_pred(p_home=0.55, p_draw=0.30, p_away=0.15)
    odds = [
        OddsLine("1x2", "home", 1.95),  # edge ~7%
        OddsLine("1x2", "draw", 4.0),   # edge 20%
    ]
    bets = detect_value(
        match_id=1, home_team="A", away_team="B", league="X",
        prediction=pred, odds_lines=odds, bankroll=1000,
    )
    # Both should pass (between 5% and 30%); draw should be first because higher edge
    assert len(bets) == 2
    assert bets[0].selection == "draw"
