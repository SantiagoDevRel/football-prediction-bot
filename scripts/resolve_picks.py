"""Auto-resolve open paper picks whose match has finished.

Usage:
    python scripts/resolve_picks.py            # resolve all + send Telegram
    python scripts/resolve_picks.py --silent   # resolve, no Telegram

Designed to run hourly via cron / Windows Task Scheduler.
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notifications.telegram_bot import send_message  # noqa: E402
from src.tracking.auto_resolver import auto_resolve_paper_picks  # noqa: E402
from src.tracking.pick_logger import compute_rolling_metrics, get_current_bankroll  # noqa: E402


def _humanize(p: dict) -> str:
    market, sel = p["market"], p["selection"]
    home, away = p["home_name"], p["away_name"]
    score = f"{p['home_goals']}-{p['away_goals']}"
    if market == "1x2":
        action = {"home": f"Gana {home}", "draw": "Empate", "away": f"Gana {away}"}.get(sel, market)
    elif market == "btts":
        action = "BTTS Sí" if sel == "yes" else "BTTS No"
    elif market.startswith("ou_"):
        line = market.removeprefix("ou_")
        action = f"Más de {line} goles" if sel == "over" else f"Menos de {line} goles"
    else:
        action = f"{market}:{sel}"
    icon = "🟢" if p["won"] else "🔴"
    status = "GANADA" if p["won"] else "PERDIDA"
    if p["won"]:
        net = float(p["stake"]) * (float(p["odds_taken"]) - 1)
    else:
        net = -float(p["stake"])
    return (
        f"{icon} <b>Pick #{p['id']} {status}</b>\n"
        f"{home} {score} {away}\n"
        f"<i>{action} @ {p['odds_taken']:.2f} · stake ${p['stake']:,.0f}</i>\n"
        f"<b>P&amp;L: ${net:+,.0f}</b>"
    )


async def main_async(silent: bool) -> None:
    resolved = await auto_resolve_paper_picks()
    if not resolved:
        print("No picks to resolve.")
        return
    print(f"Resolved {len(resolved)} picks.")

    if silent:
        return

    bankroll = get_current_bankroll("paper")
    metrics = compute_rolling_metrics("paper", days=30)
    parts = [
        "<b>🎯 Resoluciones automáticas</b>",
        f"<i>Bankroll: ${bankroll:,.0f} · {len(resolved)} picks cerradas</i>",
        "",
    ]
    parts.extend(_humanize(p) for p in resolved)
    parts.append("")
    parts.append(
        f"<b>Últimos 30 días:</b> {metrics['n']} resueltas · "
        f"win rate {metrics['win_rate']:.0%} · "
        f"ROI {metrics['roi']:+.1%} · "
        f"P&amp;L ${metrics['total_pnl']:+,.0f}"
    )
    await send_message("\n".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true", help="don't send Telegram message")
    args = parser.parse_args()
    asyncio.run(main_async(args.silent))
