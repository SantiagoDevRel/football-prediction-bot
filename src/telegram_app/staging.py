"""Stage value-bet candidates per chat_id, retrieve by session number.

Lifecycle:
    /picks  -> clear_chat() + stage_picks(...) writes a fresh batch numbered 1..N
    /aposte -> get_staged(chat_id, n) returns the candidate
            -> log_pick(...) promotes it to the picks table (real)
            -> staged row stays for /historial visibility but is no longer active
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.data.persist import get_conn


@dataclass
class StagedPick:
    chat_id: str
    session_number: int
    match_id: int
    home_team: str
    away_team: str
    league: str
    market: str
    selection: str
    odds: float
    bookmaker: str
    model_probability: float
    fair_odds: float
    edge: float
    confidence: float | None
    recommended_stake: float
    reasoning: str
    kickoff_utc: str | None


def clear_chat(chat_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM staged_picks WHERE chat_id = ?", (chat_id,))


def stage_picks(chat_id: str, value_bets: list[dict]) -> list[StagedPick]:
    """Insert N picks numbered 1..N. Returns the StagedPicks for display."""
    clear_chat(chat_id)
    out: list[StagedPick] = []
    with get_conn() as conn:
        for i, vb in enumerate(value_bets, start=1):
            conn.execute(
                """
                INSERT INTO staged_picks
                    (chat_id, session_number, match_id, home_team, away_team, league,
                     market, selection, odds, bookmaker, model_probability, fair_odds,
                     edge, confidence, recommended_stake, reasoning, kickoff_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id, i, vb["match_id"], vb["home_team"], vb["away_team"],
                    vb["league"], vb["market"], vb["selection"], vb["odds"],
                    vb["bookmaker"], vb["model_probability"], vb["fair_odds"],
                    vb["edge"], vb.get("confidence"), vb["recommended_stake"],
                    vb.get("reasoning", ""), vb.get("kickoff", ""),
                ),
            )
            out.append(StagedPick(
                chat_id=chat_id, session_number=i,
                match_id=vb["match_id"], home_team=vb["home_team"],
                away_team=vb["away_team"], league=vb["league"],
                market=vb["market"], selection=vb["selection"],
                odds=vb["odds"], bookmaker=vb["bookmaker"],
                model_probability=vb["model_probability"],
                fair_odds=vb["fair_odds"], edge=vb["edge"],
                confidence=vb.get("confidence"),
                recommended_stake=vb["recommended_stake"],
                reasoning=vb.get("reasoning", ""),
                kickoff_utc=vb.get("kickoff"),
            ))
    logger.info(f"staged {len(out)} picks for chat {chat_id}")
    return out


def get_staged(chat_id: str, session_number: int) -> StagedPick | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM staged_picks WHERE chat_id = ? AND session_number = ?",
            (chat_id, session_number),
        ).fetchone()
    if row is None:
        return None
    return StagedPick(
        chat_id=row["chat_id"], session_number=row["session_number"],
        match_id=row["match_id"], home_team=row["home_team"],
        away_team=row["away_team"], league=row["league"],
        market=row["market"], selection=row["selection"],
        odds=row["odds"], bookmaker=row["bookmaker"],
        model_probability=row["model_probability"],
        fair_odds=row["fair_odds"], edge=row["edge"],
        confidence=row["confidence"], recommended_stake=row["recommended_stake"],
        reasoning=row["reasoning"] or "", kickoff_utc=row["kickoff_utc"],
    )
