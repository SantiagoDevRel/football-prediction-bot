"""The Odds API client — aggregates odds from many bookmakers in one call.

Service: https://the-odds-api.com
Free tier: 500 requests/month. Enough for once-daily polling of our two
leagues (~30 fixtures × 2 markets × 1 call/day = 60 req/day → 1800/month).
We cache results and only refresh when needed.

Returns the BEST price across all monitored bookmakers per (match, market,
selection). The detector uses that best price for value calculations,
which gives the user the highest possible CLV automatically.

Setup: add ODDS_API_KEY to .env (sign up free at the-odds-api.com).
Without a key, this module is a no-op (returns empty list) and the rest
of the pipeline keeps working with Wplay-only odds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from loguru import logger


ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Map our internal league slug -> The Odds API sport_key.
# NOTE: the-odds-api does NOT cover Liga BetPlay (Colombia Primera A) — for
# BetPlay matches we rely on Wplay-only odds, but form/H2H from our DB still
# enrich Claude's reasoning.
SPORT_KEY_BY_SLUG = {
    "premier_league":   "soccer_epl",
    "champions_league": "soccer_uefa_champs_league",
    "libertadores":     "soccer_conmebol_copa_libertadores",
    "sudamericana":     "soccer_conmebol_copa_sudamericana",
    # liga_betplay: not supported on the-odds-api
}


@dataclass
class MultiBookieOdds:
    league_slug: str
    home_team: str
    away_team: str
    commence_time: datetime
    market: str          # "1x2" | "ou_2.5" | etc. (btts NOT free-tier)
    selection: str       # "home" | "draw" | "away" | "over" | "under"
    best_odds: float
    best_bookmaker: str  # name of the casa with the best price
    casas_seen: int      # how many bookmakers offered this market
    # Enriched stats for Claude reasoning:
    median_odds: float = 0.0   # market consensus
    worst_odds: float = 0.0    # the lowest price offered
    all_prices: list[float] | None = None  # full distribution


def _market_label(api_market: str) -> str:
    """Translate The Odds API market key to our internal market name."""
    return {
        "h2h": "1x2",
        "totals": "ou_2.5",  # default; we filter by point=2.5 below
        "btts": "btts",
    }.get(api_market, api_market)


async def fetch_multi_bookie_odds(
    league_slug: str, api_key: str, regions: str = "uk,eu,us",
    markets: str = "h2h,totals",
) -> list[MultiBookieOdds]:
    """Hit /v4/sports/{sport_key}/odds with the given markets.

    Free tier supports h2h (1X2) + totals (O/U) on most sports. BTTS is paid.

    Returns full consensus stats (best, median, worst, all prices) per
    (match, market, selection). Includes pre-match AND in-play if a match
    is live when called — the API tags both with the same payload format.
    """
    sport_key = SPORT_KEY_BY_SLUG.get(league_slug)
    if not sport_key:
        return []
    if not api_key:
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            remaining = resp.headers.get("X-Requests-Remaining", "?")
            logger.info(f"odds-api {league_slug}: {len(data)} events, quota remaining: {remaining}")
    except httpx.HTTPError as exc:
        logger.warning(f"odds-api {league_slug} failed: {exc}")
        return []

    out: list[MultiBookieOdds] = []
    for event in data:
        commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        home = event["home_team"]
        away = event["away_team"]
        # (market, selection) -> {bookie_name: price}
        prices_per_sel: dict[tuple[str, str], dict[str, float]] = {}

        for bm in event.get("bookmakers", []):
            bookie_name = bm.get("title", bm.get("key", "?"))
            for mkt in bm.get("markets", []):
                api_market = mkt["key"]
                for outcome in mkt.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = float(outcome.get("price", 0))
                    if price <= 1.01:
                        continue

                    if api_market == "h2h":
                        if name == home:
                            sel = "home"
                        elif name == away:
                            sel = "away"
                        elif name.lower() in ("draw", "tie"):
                            sel = "draw"
                        else:
                            continue
                        market_key = "1x2"
                    elif api_market == "totals":
                        # totals comes with point per outcome
                        point = outcome.get("point")
                        # Map any half-line to ou_X.5 internal key
                        if point in (1.5, 2.5, 3.5):
                            market_key = f"ou_{point}"
                        else:
                            continue
                        sel = "over" if name.lower() == "over" else "under"
                    else:
                        continue

                    key = (market_key, sel)
                    prices_per_sel.setdefault(key, {})[bookie_name] = price

        for (market_key, sel), book_prices in prices_per_sel.items():
            if not book_prices:
                continue
            sorted_prices = sorted(book_prices.values())
            n = len(sorted_prices)
            median_p = sorted_prices[n // 2] if n % 2 == 1 else (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2
            best_price = max(book_prices.values())
            best_bm = max(book_prices.items(), key=lambda kv: kv[1])[0]
            out.append(MultiBookieOdds(
                league_slug=league_slug,
                home_team=home, away_team=away,
                commence_time=commence,
                market=market_key, selection=sel,
                best_odds=best_price, best_bookmaker=best_bm,
                casas_seen=n,
                median_odds=median_p,
                worst_odds=min(book_prices.values()),
                all_prices=sorted_prices,
            ))

    return out


# Quota helper

async def fetch_remaining_quota(api_key: str) -> dict | None:
    """X-Requests-Remaining header is returned with every request. We check
    it via a minimal GET to /sports."""
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key})
            resp.raise_for_status()
            return {
                "remaining": resp.headers.get("X-Requests-Remaining", "?"),
                "used": resp.headers.get("X-Requests-Used", "?"),
                "last_cost": resp.headers.get("X-Requests-Last", "?"),
            }
    except httpx.HTTPError:
        return None
