"""Interactive Telegram bot for the football-prediction-bot.

Long-polling listener that responds to user commands. Picks are staged
per chat_id so the user can /aposte by simple session number (1..N) and
the bot commits them to the picks table only after explicit confirmation.
"""
