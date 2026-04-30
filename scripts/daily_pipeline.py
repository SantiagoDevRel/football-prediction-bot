"""Daily pipeline (Phase 2).

Runs in this order:
    1. Pull today's fixtures from API-Football for our target leagues
    2. Pull current Wplay odds for those fixtures
    3. Run LLM feature extraction (lineups, news, injuries)
    4. Run all base models, then ensemble
    5. Detect value bets
    6. Log picks (paper or real)
    7. Push alerts to Telegram

Designed to run via cron / scheduled task. In Phase 0 it's a stub.
"""


def main() -> None:
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    main()
