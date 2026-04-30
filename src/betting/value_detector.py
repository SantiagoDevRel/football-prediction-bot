"""Compare model predictions against bookmaker odds; flag value bets.

A pick is suggested only when ALL gates pass:
    1. edge > MIN_EDGE (default 5%)
    2. odds within sane range (>1.10 and <20.00) - extreme prices are noise
    3. model probability is reasonable (>2% and <98% - extremes mean noise)
    4. confidence >= MIN_CONFIDENCE (when available; ignored if model has none)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.betting.kelly import kelly_stake, edge as edge_fn
from src.config import settings
from src.models.base import MatchProbabilities


@dataclass
class OddsLine:
    market: str       # "1x2" | "ou_2.5" | "btts"
    selection: str    # "home" | "draw" | "away" | "over" | "under" | "yes" | "no"
    odds: float
    bookmaker: str = "wplay"


@dataclass
class ValueBet:
    match_id: int
    home_team: str
    away_team: str
    league: str
    market: str
    selection: str
    odds: float
    bookmaker: str
    model_probability: float
    fair_odds: float
    edge: float
    confidence: float | None
    recommended_stake: float
    reasoning: str = ""


def _select_prob(prediction: MatchProbabilities, market: str, selection: str) -> float | None:
    if market == "1x2":
        return {"home": prediction.p_home_win,
                "draw": prediction.p_draw,
                "away": prediction.p_away_win}.get(selection)
    if market == "ou_2.5":
        return {"over": prediction.p_over_2_5,
                "under": prediction.p_under_2_5}.get(selection)
    if market == "btts":
        return {"yes": prediction.p_btts_yes,
                "no": prediction.p_btts_no}.get(selection)
    return None


def detect_value(
    *,
    match_id: int,
    home_team: str,
    away_team: str,
    league: str,
    prediction: MatchProbabilities,
    odds_lines: list[OddsLine],
    bankroll: float,
    min_edge: float | None = None,
    min_confidence: float | None = None,
    kelly_fraction: float | None = None,
) -> list[ValueBet]:
    """Scan all (market, selection) combinations and return value bets above threshold."""
    min_edge = settings.min_edge if min_edge is None else min_edge
    min_confidence = settings.min_confidence if min_confidence is None else min_confidence
    kelly_fraction = settings.kelly_fraction if kelly_fraction is None else kelly_fraction

    out: list[ValueBet] = []
    for line in odds_lines:
        if not (1.10 < line.odds < 20.0):
            continue
        prob = _select_prob(prediction, line.market, line.selection)
        if prob is None:
            continue
        if not (0.02 < prob < 0.98):
            continue
        e = edge_fn(line.odds, prob)
        if e < min_edge:
            continue
        if prediction.confidence is not None and prediction.confidence < min_confidence:
            continue

        stake = kelly_stake(
            bankroll=bankroll, odds=line.odds, probability=prob,
            fraction=kelly_fraction, max_stake_pct=0.05,
        )
        if stake <= 0:
            continue

        fair = 1.0 / prob
        out.append(ValueBet(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            market=line.market,
            selection=line.selection,
            odds=line.odds,
            bookmaker=line.bookmaker,
            model_probability=prob,
            fair_odds=fair,
            edge=e,
            confidence=prediction.confidence,
            recommended_stake=stake,
            reasoning=(
                f"Model: {prob:.1%} | Wplay: {line.odds:.2f} | Fair: {fair:.2f} "
                f"| Edge: +{e:.1%}"
            ),
        ))
    # Sort by edge desc — most attractive first
    out.sort(key=lambda v: v.edge, reverse=True)
    return out
