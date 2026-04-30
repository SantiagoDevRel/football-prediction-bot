"""Backtesting harness.

Replays historical seasons match-by-match in chronological order. For each match:
    - Generates predictions using only data available BEFORE that match (no leakage)
    - Compares against historical odds (closing if available, else best snapshot)
    - Computes hypothetical picks, CLV, ROI

Output: per-model and per-market metrics, calibration plots, equity curve.

Phase 1 implementation.
"""


def run_backtest(
    league: str,
    seasons: list[int],
    models: list[str],
    initial_bankroll: float = 1_000_000.0,
) -> dict:
    raise NotImplementedError("Phase 1")


if __name__ == "__main__":
    run_backtest("EPL", [2022, 2023, 2024], ["dixon_coles", "elo"])
