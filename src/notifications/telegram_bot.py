"""Telegram bot for push notifications.

Sends value-bet alerts to a chat. User reads, decides, and places the bet
manually on Wplay.

Phase 2 implementation.
"""
from __future__ import annotations

from src.betting.value_detector import ValueBet


async def send_pick_alert(pick: ValueBet, bot_token: str, chat_id: str) -> None:
    """Format and send a value bet to Telegram.

    Message format:
        ⭐ <home> vs <away> (<league>)
        Mercado: <market> · <selection>
        Cuota Wplay: <odds>  (Fair: <fair>)
        Edge: +<edge>%  ·  Confianza: <conf>%
        Stake recomendado: $<stake> (¼ Kelly)
        ──
        <reasoning>
    """
    raise NotImplementedError("Phase 2")


async def send_daily_summary(
    bot_token: str,
    chat_id: str,
    n_picks: int,
    bankroll: float,
    rolling_clv: float,
    rolling_roi: float,
) -> None:
    raise NotImplementedError("Phase 2")
