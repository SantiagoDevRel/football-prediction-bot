"""Conversational agent for the Telegram bot.

Wraps Claude Sonnet 4.6 with a tool-use loop that gives the model access to
the bot's existing capabilities (picks, balance, history, match analysis,
custom-bet logging). The slash commands stay as fast paths; this is for
free-form conversation, parlay-slip pastes, and multi-turn reasoning.
"""
from src.agent.conversational import ConversationalAgent  # noqa: F401
