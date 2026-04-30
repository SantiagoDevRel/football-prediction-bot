"""Telegram notifications.

Sends a single text message with a list of value bets / predictions.
If TELEGRAM_BOT_TOKEN is not set, falls back to a console print so the
pipeline still works end-to-end without manual setup.
"""
from __future__ import annotations

import httpx
from loguru import logger

from src.config import settings


async def _send_telegram_message(text: str) -> bool:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            })
            resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning(f"telegram send failed: {exc}")
        return False


async def send_message(text: str) -> None:
    """Send via Telegram if configured, otherwise print to console."""
    sent = await _send_telegram_message(text)
    if not sent:
        # Strip HTML tags for console fallback
        import re
        clean = re.sub(r"<[^>]+>", "", text)
        print("\n" + "=" * 60)
        print("(Telegram not configured — printing to console)")
        print("=" * 60)
        print(clean)
        print("=" * 60 + "\n")
