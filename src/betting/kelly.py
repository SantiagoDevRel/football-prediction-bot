"""Kelly Criterion for bet sizing.

Full Kelly formula:
    f* = (b * p - q) / b
    where:
        b = decimal odds - 1 (net odds, e.g. odds 2.50 -> b = 1.50)
        p = our estimated probability of winning
        q = 1 - p

Fractional Kelly:
    f = fraction * f*
    fraction = 0.25 (¼ Kelly) is the standard for sports betting:
        - Reduces variance dramatically
        - Survives miscalibration of probabilities
        - Long-term growth slightly slower but much more stable
"""


def kelly_stake(
    bankroll: float,
    odds: float,
    probability: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
    min_stake: float = 0.0,
) -> float:
    """Compute recommended stake.

    Args:
        bankroll: current total bankroll
        odds: decimal odds offered by the bookmaker
        probability: our estimated win probability
        fraction: Kelly fraction (default ¼ Kelly)
        max_stake_pct: hard cap as % of bankroll (default 5%)
        min_stake: hard floor for any recommendation. If edge > 0 but the
            Kelly+cap stake comes out below min_stake, we DO NOT recommend
            a smaller bet — we return 0 so the caller drops the pick.
            (User rule: "no recommendations below the floor; if the bankroll
            can't honor the floor inside the cap, don't recommend at all.")

    Returns:
        Recommended stake amount (>= 0). Returns 0 if no edge OR if the
        floor cannot be honored without exceeding max_stake_pct of bankroll.
    """
    if odds <= 1.0 or not (0.0 < probability < 1.0):
        return 0.0

    b = odds - 1.0
    q = 1.0 - probability
    f_star = (b * probability - q) / b

    if f_star <= 0:
        return 0.0

    stake_fraction = min(f_star * fraction, max_stake_pct)
    stake = bankroll * stake_fraction
    if min_stake > 0 and stake < min_stake:
        # Floor is hard. If the cap won't allow it, refuse the pick.
        if bankroll * max_stake_pct < min_stake:
            return 0.0
        stake = min_stake
    return stake


def edge(odds: float, probability: float) -> float:
    """Edge = (probability * odds) - 1. Positive means +EV."""
    return probability * odds - 1.0
