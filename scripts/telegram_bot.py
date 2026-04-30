"""Long-polling Telegram bot entry point.

Usage:
    .venv\\Scripts\\python.exe scripts\\telegram_bot.py

Runs forever, reads commands from your Telegram chat, replies. Stop with Ctrl+C.
For background execution, use scripts\\run_bot.cmd.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402
from telegram.ext import (  # noqa: E402
    Application, CallbackQueryHandler, CommandHandler,
)

from src.config import settings  # noqa: E402
from src.telegram_app.handlers import (  # noqa: E402
    callback_handler,
    cmd_analizar,
    cmd_aposte,
    cmd_balance,
    cmd_help,
    cmd_historial,
    cmd_picks,
    cmd_resolver,
    cmd_resolver_auto,
    cmd_start,
)


def main() -> None:
    if not settings.telegram_bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env. Aborting.")
        sys.exit(1)

    logger.info("Starting Telegram bot (long polling)…")
    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("picks", cmd_picks))
    app.add_handler(CommandHandler("aposte", cmd_aposte))
    app.add_handler(CommandHandler("resolver", cmd_resolver))
    app.add_handler(CommandHandler("resolver_auto", cmd_resolver_auto))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("historial", cmd_historial))
    app.add_handler(CommandHandler("analizar", cmd_analizar))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
