"""In-play v0: condition pre-match Dixon-Coles output on current state.

Honest scope:
    This is NOT a model trained on minute-by-minute data. It's a mathematical
    re-computation: given pre-match λ (home rate) and μ (away rate), and given
    the current score + minutes elapsed, predict the remainder using a Poisson
    on the time-fraction left.

Limitations (be honest with users):
    - Doesn't account for red cards, fatigue, score-dependent tactics
      (e.g. "team protecting a lead plays more conservatively").
    - λ and μ stay flat throughout the match - in reality goal rates vary
      with stage of match.
    - Doesn't update on shots, possession, expected goals so far.

Despite that, it's still better than using the pre-match probabilities
unchanged. If a match is 2-0 at minute 80, the remaining 10 minutes can
only see 0-2 more goals roughly, so Over 2.5 is GUARANTEED already and
BTTS is impossible if home didn't concede yet.
"""
from __future__ import annotations

from math import exp, factorial

import numpy as np

from src.models.base import MatchProbabilities


_CAP = 7  # max additional goals to consider (covers 99.99% of remaining-game possibilities)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def condition_on_state(
    pre_match: MatchProbabilities,
    current_home: int,
    current_away: int,
    minute: int,
    full_match_minutes: int = 90,
) -> MatchProbabilities:
    """Recompute markets given the current state.

    Args:
        pre_match: probabilities from Dixon-Coles or ensemble for the WHOLE match.
                   We only need expected_home_goals and expected_away_goals.
        current_home, current_away: goals scored so far.
        minute: current match minute (0..120).
        full_match_minutes: 90 in regulation. Pass 120 if extra time is in play.

    Returns:
        New MatchProbabilities for the remainder + final state. The returned
        object's expected_*_goals are TOTAL (current + remainder), so 1X2/OU/BTTS
        markets compute against the final score.
    """
    if minute >= full_match_minutes:
        # Match over - no more goals expected. Build deterministic distribution.
        return _deterministic_at_final(current_home, current_away, pre_match)

    # Time-remaining fraction (defensive clamp)
    fraction_left = max(0.0, (full_match_minutes - minute) / float(full_match_minutes))
    remaining_lambda = pre_match.expected_home_goals * fraction_left
    remaining_mu = pre_match.expected_away_goals * fraction_left

    # Distribution of (home_extra, away_extra) goals in the remaining time
    sm_extra = np.zeros((_CAP + 1, _CAP + 1))
    for i in range(_CAP + 1):
        for j in range(_CAP + 1):
            sm_extra[i, j] = _poisson_pmf(i, remaining_lambda) * _poisson_pmf(j, remaining_mu)
    s = sm_extra.sum()
    if s > 0:
        sm_extra = sm_extra / s

    # Final-score grid: shift by current
    h_extra, a_extra = np.indices(sm_extra.shape)
    final_h = h_extra + current_home
    final_a = a_extra + current_away
    total_final = final_h + final_a
    margin_final = final_h - final_a

    # 1X2
    p_home_win = float(sm_extra[margin_final > 0].sum())
    p_draw = float(sm_extra[margin_final == 0].sum())
    p_away_win = float(sm_extra[margin_final < 0].sum())

    # Over/Under thresholds
    p_over_1_5 = float(sm_extra[total_final > 1].sum())
    p_over_2_5 = float(sm_extra[total_final > 2].sum())
    p_over_3_5 = float(sm_extra[total_final > 3].sum())

    # BTTS: both teams need >= 1. We know current scores; compute
    # P(home_final >= 1 and away_final >= 1).
    p_home_scored = current_home > 0 or remaining_lambda > 0
    p_away_scored = current_away > 0 or remaining_mu > 0
    if current_home > 0 and current_away > 0:
        p_btts_yes = 1.0  # both already scored
    elif current_home > 0:
        # Need away to score at least once in remainder
        p_btts_yes = 1.0 - _poisson_pmf(0, remaining_mu)
    elif current_away > 0:
        p_btts_yes = 1.0 - _poisson_pmf(0, remaining_lambda)
    else:
        # Neither has scored yet
        p_btts_yes = (1.0 - _poisson_pmf(0, remaining_lambda)) * (1.0 - _poisson_pmf(0, remaining_mu))
    p_btts_no = 1.0 - p_btts_yes

    # Handicap -1.5: home wins by 2+
    p_home_minus_1_5 = float(sm_extra[margin_final >= 2].sum())

    return MatchProbabilities(
        p_home_win=p_home_win,
        p_draw=p_draw,
        p_away_win=p_away_win,
        p_over_2_5=p_over_2_5,
        p_under_2_5=float(1.0 - p_over_2_5),
        p_over_1_5=p_over_1_5,
        p_under_1_5=float(1.0 - p_over_1_5),
        p_over_3_5=p_over_3_5,
        p_under_3_5=float(1.0 - p_over_3_5),
        p_btts_yes=p_btts_yes,
        p_btts_no=p_btts_no,
        p_home_minus_1_5=p_home_minus_1_5,
        p_away_plus_1_5=float(1.0 - p_home_minus_1_5),
        expected_home_goals=current_home + remaining_lambda,
        expected_away_goals=current_away + remaining_mu,
        features={
            "model": "inplay_v0",
            "minute": minute,
            "current_home": current_home,
            "current_away": current_away,
            "remaining_lambda": remaining_lambda,
            "remaining_mu": remaining_mu,
        },
    )


def _deterministic_at_final(hg: int, ag: int, pre: MatchProbabilities) -> MatchProbabilities:
    """Match is finished — return a degenerate distribution (everything is fixed)."""
    return MatchProbabilities(
        p_home_win=1.0 if hg > ag else 0.0,
        p_draw=1.0 if hg == ag else 0.0,
        p_away_win=1.0 if ag > hg else 0.0,
        p_over_2_5=1.0 if hg + ag > 2 else 0.0,
        p_under_2_5=0.0 if hg + ag > 2 else 1.0,
        p_over_1_5=1.0 if hg + ag > 1 else 0.0,
        p_under_1_5=0.0 if hg + ag > 1 else 1.0,
        p_over_3_5=1.0 if hg + ag > 3 else 0.0,
        p_under_3_5=0.0 if hg + ag > 3 else 1.0,
        p_btts_yes=1.0 if hg > 0 and ag > 0 else 0.0,
        p_btts_no=0.0 if hg > 0 and ag > 0 else 1.0,
        p_home_minus_1_5=1.0 if hg - ag >= 2 else 0.0,
        p_away_plus_1_5=0.0 if hg - ag >= 2 else 1.0,
        expected_home_goals=float(hg),
        expected_away_goals=float(ag),
        features={"model": "inplay_v0", "status": "finished", "current_home": hg, "current_away": ag},
    )
