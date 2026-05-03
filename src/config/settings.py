from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    # ESPN public scoreboard requires no auth → no key here
    # football-data.co.uk CSVs are public → no key here
    anthropic_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    # Optional: the-odds-api.com key for multi-bookmaker comparison.
    # Without this, the system runs Wplay-only (still functional).
    odds_api_key: str = ""
    # Optional: api-sports.io v3 football key. Provides lineups, injuries,
    # and BetPlay multi-bookmaker odds (where the-odds-api doesn't reach).
    # Free tier: 100 requests/day. Sign up at https://dashboard.api-football.com/
    api_football_key: str = ""

    # Operation mode
    betting_mode: Literal["paper", "real"] = "paper"

    # Bankroll & risk
    paper_bankroll_initial: float = 100_000.0
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    min_edge: float = Field(default=0.05, ge=0.0, le=1.0)
    # Edges above this are almost always model errors, not real market value.
    # Filter them out automatically so we don't bet "too good to be true" picks.
    max_edge: float = Field(default=0.30, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)

    # Storage
    database_url: str = f"sqlite:///{ROOT_DIR / 'data' / 'db' / 'football_bot.db'}"

    # Logging
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.removeprefix("sqlite:///"))
        raise ValueError("db_path only available for sqlite URLs")

    @property
    def is_paper_mode(self) -> bool:
        return self.betting_mode == "paper"


settings = Settings()
