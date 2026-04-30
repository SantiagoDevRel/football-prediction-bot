"""Initialize SQLite schema for the prediction bot.

Run once to create tables. Idempotent: re-running won't drop data, only adds missing tables.

Usage:
    python scripts/init_db.py
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402

SCHEMA = """
-- Leagues we track. Premier League and Liga BetPlay to start.
CREATE TABLE IF NOT EXISTS leagues (
    id           INTEGER PRIMARY KEY,
    api_id       INTEGER UNIQUE,           -- API-Football league id
    name         TEXT NOT NULL,
    country      TEXT NOT NULL,
    season       INTEGER NOT NULL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teams (
    id           INTEGER PRIMARY KEY,
    api_id       INTEGER UNIQUE,
    name         TEXT NOT NULL,
    short_name   TEXT,
    country      TEXT,
    league_id    INTEGER REFERENCES leagues(id),
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Historical and upcoming fixtures
CREATE TABLE IF NOT EXISTS matches (
    id             INTEGER PRIMARY KEY,
    api_id         INTEGER UNIQUE,
    league_id      INTEGER REFERENCES leagues(id),
    season         INTEGER NOT NULL,
    home_team_id   INTEGER REFERENCES teams(id),
    away_team_id   INTEGER REFERENCES teams(id),
    kickoff_utc    TEXT NOT NULL,
    status         TEXT NOT NULL,             -- scheduled / live / finished
    home_goals     INTEGER,
    away_goals     INTEGER,
    home_xg        REAL,
    away_xg        REAL,
    home_lineup    TEXT,                       -- JSON
    away_lineup    TEXT,                       -- JSON
    referee        TEXT,
    venue          TEXT,
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at     TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches(kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);

-- Per-match boxscore stats (yellow/red cards, corners, fouls, shots)
-- Backfilled from ESPN summary endpoint via scripts/backfill_boxscore.py.
CREATE TABLE IF NOT EXISTS match_stats (
    match_id          INTEGER PRIMARY KEY REFERENCES matches(id),
    home_yellow_cards INTEGER,
    away_yellow_cards INTEGER,
    home_red_cards    INTEGER,
    away_red_cards    INTEGER,
    home_corners      INTEGER,
    away_corners      INTEGER,
    home_fouls        INTEGER,
    away_fouls        INTEGER,
    home_shots        INTEGER,
    away_shots        INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_possession   REAL,
    away_possession   REAL,
    fetched_at        TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Odds snapshots (one row per snapshot per market per match per bookie)
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id            INTEGER PRIMARY KEY,
    match_id      INTEGER REFERENCES matches(id),
    bookmaker     TEXT NOT NULL,               -- "wplay", "bet365", etc.
    market        TEXT NOT NULL,               -- "1x2", "ou_2.5", "btts", "ah_-1"
    selection     TEXT NOT NULL,               -- "home", "draw", "away", "over", "yes"
    odds          REAL NOT NULL,
    is_closing    INTEGER DEFAULT 0,           -- 1 if captured at kickoff
    captured_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_odds_match_market ON odds_snapshots(match_id, market, selection);
CREATE INDEX IF NOT EXISTS idx_odds_closing ON odds_snapshots(is_closing);

-- Predictions: one row per (match, market, selection, model)
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY,
    match_id      INTEGER REFERENCES matches(id),
    market        TEXT NOT NULL,
    selection     TEXT NOT NULL,
    model         TEXT NOT NULL,               -- "dixon_coles", "elo", "xgboost", "ensemble"
    probability   REAL NOT NULL,
    confidence    REAL,                        -- model-specific uncertainty (Bayesian)
    features_used TEXT,                        -- JSON
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pred_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model);

-- LLM-generated qualitative features
CREATE TABLE IF NOT EXISTS qualitative_features (
    id           INTEGER PRIMARY KEY,
    match_id     INTEGER REFERENCES matches(id),
    flags        TEXT NOT NULL,                -- JSON list, e.g. ["injuries:home:3", "rotation:away"]
    summary      TEXT,                          -- short text from LLM
    raw_news     TEXT,                          -- raw input (truncated)
    model_used   TEXT NOT NULL,                 -- "claude-opus-4-7", etc.
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Picks: actual decisions to bet (paper or real)
CREATE TABLE IF NOT EXISTS picks (
    id                INTEGER PRIMARY KEY,
    match_id          INTEGER REFERENCES matches(id),
    market            TEXT NOT NULL,
    selection         TEXT NOT NULL,
    odds_taken        REAL NOT NULL,
    bookmaker         TEXT NOT NULL,
    model_probability REAL NOT NULL,
    edge              REAL NOT NULL,
    confidence        REAL,
    stake             REAL NOT NULL,
    mode              TEXT NOT NULL,            -- "paper" or "real"
    placed_at         TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Resolution (filled after match)
    closing_odds      REAL,                     -- for CLV calc
    won               INTEGER,                  -- 0/1, NULL until resolved
    payout            REAL,                     -- net P&L in COP
    clv               REAL,                     -- closing line value
    resolved_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_picks_mode ON picks(mode);
CREATE INDEX IF NOT EXISTS idx_picks_resolved ON picks(won);

-- Bankroll history (one row per change: stake, payout, deposit, withdrawal)
CREATE TABLE IF NOT EXISTS bankroll_history (
    id          INTEGER PRIMARY KEY,
    mode        TEXT NOT NULL,                  -- "paper" or "real"
    pick_id     INTEGER REFERENCES picks(id),   -- NULL for manual deposits
    delta       REAL NOT NULL,                  -- + or -
    balance     REAL NOT NULL,                  -- balance after this delta
    note        TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Staged picks for the Telegram bot. The bot runs the pipeline, stages
-- candidates here numbered 1..N per chat_id. When the user runs /aposte N
-- we promote that stage row to a real picks row.
CREATE TABLE IF NOT EXISTS staged_picks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id           TEXT NOT NULL,
    session_number    INTEGER NOT NULL,
    match_id          INTEGER REFERENCES matches(id),
    home_team         TEXT,
    away_team         TEXT,
    league            TEXT,
    market            TEXT NOT NULL,
    selection         TEXT NOT NULL,
    odds              REAL NOT NULL,
    bookmaker         TEXT NOT NULL,
    model_probability REAL NOT NULL,
    fair_odds         REAL,
    edge              REAL,
    confidence        REAL,
    recommended_stake REAL,
    reasoning         TEXT,
    kickoff_utc       TEXT,
    created_at        TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_staged_chat ON staged_picks(chat_id, session_number);

-- Model performance tracking (rollups)
CREATE TABLE IF NOT EXISTS model_performance (
    id              INTEGER PRIMARY KEY,
    model           TEXT NOT NULL,
    league_id       INTEGER REFERENCES leagues(id),
    market          TEXT NOT NULL,
    period_start    TEXT NOT NULL,
    period_end      TEXT NOT NULL,
    n_predictions   INTEGER NOT NULL,
    log_loss        REAL,
    brier_score     REAL,
    win_rate        REAL,
    avg_clv         REAL,
    roi             REAL,
    computed_at     TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db() -> None:
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        n_tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        print(f"[OK] DB initialized at {db_path}")
        print(f"[OK] Tables present: {n_tables}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
