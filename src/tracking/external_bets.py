"""Register bets the user placed manually outside the /picks flow.

The typical flow:
    1. User pastes a Wplay confirmation block (or describes bets in prose).
    2. Claude extracts a list of {match, odds, stake, optional selection hint}.
    3. For each parsed bet we:
         a. Match home/away to teams in our DB (fuzzy)
         b. Find the upcoming match
         c. If selection is ambiguous (Wplay's "Ganador X v Y" doesn't say
            which side), infer by comparing the user's odds against the
            current Wplay 1X2 odds and pick whichever is closest.
         d. Insert into picks with mode='real' (or 'paper'), model_probability=0,
            edge=0, bypassing risk gates (the bet is already placed).
    4. The auto-resolver picks them up like any other pick when the match ends.

Limitations (acknowledged):
    - Selection inference via "closest odds" is heuristic. If user bet a
      market we don't scrape (handicap, exact score), we can't disambiguate.
    - Match name fuzzy matching can fail for unusual spellings.
    Both failure modes return a structured error so the bot can ask for
    clarification.
"""
from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from anthropic import AsyncAnthropic
from loguru import logger

from src.betting.value_detector import ValueBet
from src.data.persist import get_conn


Mode = Literal["paper", "real"]


@dataclass
class ParsedBet:
    """One bet extracted from raw user text."""
    home_team: str
    away_team: str
    odds: float
    stake: float
    market: str = "1x2"           # if Claude can tell; defaults to 1x2
    selection_hint: str | None = None  # 'home' | 'away' | 'draw' | etc, if explicit
    raw: str = ""                 # original text fragment, for debugging


@dataclass
class ResolvedBet:
    """A ParsedBet matched to a DB row and a concrete selection."""
    match_id: int
    home_team: str           # canonical name from DB
    away_team: str
    league: str
    market: str
    selection: str
    odds: float
    stake: float
    confidence: float        # 0..1, how sure we are about the selection
    note: str = ""           # any caveats for the user


@dataclass
class RegistrationResult:
    inserted: list[tuple[int, ResolvedBet]] = field(default_factory=list)  # (pick_id, bet)
    skipped: list[tuple[ParsedBet, str]] = field(default_factory=list)     # (bet, reason)


# ---------- Parsing ----------

PARSE_SYSTEM_PROMPT = """Eres un parser de boletas de apuestas deportivas Wplay.

El usuario pega el texto completo de su confirmación. Cada apuesta puede aparecer en formato resumido o expandido. Extrae una entrada por apuesta. El formato expandido SÍ dice exactamente qué seleccionó.

Formato resumido (cabecera Wplay):
    Ganador    Chelsea v Nottingham Forest
    $50,000
    4.10

Formato expandido (clave: dice exactamente la selección):
    Fecha y Hora del evento  Descripción del evento  Apuesta  Selección  ...
    04 Mayo 09:00  Chelsea v Nottingham Forest  Resultado Tiempo Completo  Empate
    $205,000  -

Si está el formato expandido, USALO para la selección. El "Ganador" del resumen es solo el nombre de columna, NO significa que apostó al ganador.

Mapeo de "Descripción del evento" → market:
- "Resultado Tiempo Completo" / "Resultado del Partido" → market="1x2"
- "Total Goles Más/Menos de" / "Total de goles" → market depende de la línea:
    "Menos de (1.5)" o "Más de (1.5)" → market="ou_1.5"
    "Menos de (2.5)" o "Más de (2.5)" → market="ou_2.5"
    "Menos de (3.5)" o "Más de (3.5)" → market="ou_3.5"
- "Ambos equipos anotan" / "Goles de ambos equipos" → market="btts"

Mapeo de "Selección" → selection_hint:
- "Empate" → "draw"
- Nombre del equipo local → "home"
- Nombre del equipo visitante → "away"
- "Más de" → "over"
- "Menos de" → "under"
- "Sí" / "Si" → "yes"
- "No" → "no"

Si solo tienes el formato resumido (sin expandido), market="1x2" y selection_hint=null (ambiguo).

Reglas:
1. Una apuesta puede aparecer dos veces sobre el mismo partido (mismo home/away pero diferente market o selección). Son DOS entradas.
2. Stakes: extraé el número, sin comas ni símbolos. "$50,000" → 50000. Usá el monto de "Monto de Apuesta", NO "Ganancia posible" ni "Recuperar Apuesta".
3. Cuotas: número decimal. "4.10" → 4.10.
4. Ignorá totalmente los valores de "RECUPERAR APUESTA?" y "Ganancia posible" — son derivados.

Responde SOLO con el tool — sin texto extra."""

_PARSE_TOOL: dict = {
    "name": "extract_bets",
    "description": "Devuelve la lista estructurada de apuestas extraídas del texto.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "home_team": {"type": "string", "description": "Nombre del equipo local."},
                        "away_team": {"type": "string", "description": "Nombre del equipo visitante."},
                        "odds": {"type": "number", "description": "Cuota decimal (ej. 4.10)."},
                        "stake": {"type": "number", "description": "Plata apostada en COP, sin símbolos."},
                        "market": {
                            "type": "string",
                            "enum": ["1x2", "btts", "ou_1.5", "ou_2.5", "ou_3.5"],
                            "description": "Mercado. Default 1x2 si solo dice 'Ganador'.",
                        },
                        "selection_hint": {
                            "type": ["string", "null"],
                            "description": "Si el texto dice explícitamente cuál seleccionó: 'home'/'draw'/'away'/'over'/'under'/'yes'/'no'. Null si ambiguo.",
                        },
                        "raw_fragment": {
                            "type": "string",
                            "description": "Texto original de esa apuesta (para debug).",
                        },
                    },
                    "required": ["home_team", "away_team", "odds", "stake", "market"],
                },
            },
        },
        "required": ["bets"],
    },
}


async def parse_pasted_text(text: str, anthropic_api_key: str) -> list[ParsedBet]:
    """Use Claude Haiku 4.5 to extract bets from raw user text."""
    client = AsyncAnthropic(api_key=anthropic_api_key)
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=[{"type": "text", "text": PARSE_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_PARSE_TOOL],
        tool_choice={"type": "tool", "name": "extract_bets"},
        messages=[{"role": "user", "content": text}],
    )
    tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
    if tool_block is None:
        logger.warning("parse_pasted_text: no tool_use in response")
        return []
    raw_bets = (tool_block.input or {}).get("bets") or []

    out: list[ParsedBet] = []
    for rb in raw_bets:
        try:
            out.append(ParsedBet(
                home_team=str(rb["home_team"]).strip(),
                away_team=str(rb["away_team"]).strip(),
                odds=float(rb["odds"]),
                stake=float(rb["stake"]),
                market=str(rb.get("market", "1x2")),
                selection_hint=(rb.get("selection_hint") or None),
                raw=str(rb.get("raw_fragment", "")),
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"skipping malformed parsed bet {rb}: {exc}")
    return out


# ---------- Match resolution ----------

def _normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _find_match(home_query: str, away_query: str) -> dict | None:
    """Find a match in DB by fuzzy team names. SQLite's LOWER doesn't strip
    diacritics, so we load all teams once and do the normalized comparison
    in Python (Independiente Medellín → independiente medellin).

    Prefers scheduled matches in next 14 days; falls back to most recent."""
    h_norm = _normalize(home_query)
    a_norm = _normalize(away_query)
    today = date.today()
    end = today + timedelta(days=14)

    with get_conn() as conn:
        team_rows = conn.execute("SELECT id, name FROM teams").fetchall()

        # Build {team_id: normalized_name}; teams whose normalized name
        # contains either query term are candidates.
        norm_by_id: dict[int, str] = {r["id"]: _normalize(r["name"]) for r in team_rows}

        def teams_matching(query: str) -> list[int]:
            return [tid for tid, nm in norm_by_id.items() if query in nm or nm in query]

        home_candidates = teams_matching(h_norm)
        away_candidates = teams_matching(a_norm)
        if not home_candidates or not away_candidates:
            return None

        # Build SQL parameter placeholders
        def _placeholders(n: int) -> str:
            return ",".join(["?"] * n)

        # First pass: try (home, away) direct + recent scheduled
        for hs, as_, swapped in [
            (home_candidates, away_candidates, False),
            (away_candidates, home_candidates, True),
        ]:
            sql = f"""
                SELECT m.id AS match_id, h.name AS home, a.name AS away,
                       l.name AS league, m.kickoff_utc, m.status,
                       m.home_team_id, m.away_team_id
                  FROM matches m
                  JOIN teams h ON m.home_team_id = h.id
                  JOIN teams a ON m.away_team_id = a.id
                  JOIN leagues l ON m.league_id = l.id
                 WHERE m.status = 'scheduled'
                   AND date(m.kickoff_utc) BETWEEN ? AND ?
                   AND m.home_team_id IN ({_placeholders(len(hs))})
                   AND m.away_team_id IN ({_placeholders(len(as_))})
                 ORDER BY m.kickoff_utc ASC LIMIT 1
            """
            row = conn.execute(
                sql, [today.isoformat(), end.isoformat(), *hs, *as_],
            ).fetchone()
            if row:
                d = dict(row)
                if swapped:
                    d["_swapped"] = True
                return d

        # Fallback: any status, ever (use direct home/away order, not swapped)
        sql = f"""
            SELECT m.id AS match_id, h.name AS home, a.name AS away,
                   l.name AS league, m.kickoff_utc, m.status,
                   m.home_team_id, m.away_team_id
              FROM matches m
              JOIN teams h ON m.home_team_id = h.id
              JOIN teams a ON m.away_team_id = a.id
              JOIN leagues l ON m.league_id = l.id
             WHERE m.home_team_id IN ({_placeholders(len(home_candidates))})
               AND m.away_team_id IN ({_placeholders(len(away_candidates))})
             ORDER BY m.kickoff_utc DESC LIMIT 1
        """
        row = conn.execute(sql, [*home_candidates, *away_candidates]).fetchone()
        if row:
            return dict(row)
    return None


def _infer_selection(market: str, user_odds: float, match_id: int,
                     hint: str | None) -> tuple[str, float, str]:
    """For a match in our DB, look up the latest scraped odds and find which
    (selection) has odds closest to the user's odds. Returns (selection,
    confidence, note)."""
    if hint:
        return hint, 1.0, "selección dada por el usuario"

    # Pull the latest 1X2 odds we scraped for this match.
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT selection, odds FROM odds_snapshots
             WHERE match_id = ? AND market = ?
             ORDER BY captured_at DESC LIMIT 10
            """,
            (match_id, market),
        ).fetchall()

    if not rows:
        # No odds context → educated guess by user_odds magnitude
        if market == "1x2":
            if user_odds <= 1.80:
                return "home", 0.4, "guess por magnitud (sin cuotas Wplay para cruzar)"
            if user_odds >= 4.0:
                return "draw", 0.3, "guess por magnitud — cuota alta sugiere empate o visitante underdog"
            return "away", 0.3, "guess por magnitud (sin cuotas Wplay para cruzar)"
        return "over", 0.4, "default (sin cuotas Wplay para cruzar)"

    # Find selection with odds closest to user's
    closest_sel = None
    closest_diff = float("inf")
    by_sel: dict[str, float] = {}
    for r in rows:
        sel = r["selection"]
        if sel in by_sel:
            continue  # take the most recent for each selection (already ordered)
        by_sel[sel] = float(r["odds"])
        diff = abs(float(r["odds"]) - user_odds)
        if diff < closest_diff:
            closest_diff = diff
            closest_sel = sel

    if closest_sel is None:
        return "home", 0.2, "no se pudo cruzar con odds Wplay"

    # Confidence based on how unique the closest match is
    second_diff = sorted(abs(v - user_odds) for v in by_sel.values())[1] if len(by_sel) > 1 else 0
    gap = second_diff - closest_diff
    confidence = min(0.95, 0.5 + gap)  # bigger gap = more confident
    note = (f"inferida por cuota (Wplay: H={by_sel.get('home','?')} "
            f"D={by_sel.get('draw','?')} A={by_sel.get('away','?')})"
            if market == "1x2" else "")
    return closest_sel, confidence, note


def resolve_bets(parsed: list[ParsedBet]) -> tuple[list[ResolvedBet], list[tuple[ParsedBet, str]]]:
    """Match each parsed bet to a DB match + concrete selection.

    Returns (resolved, errors). Errors are (parsed_bet, reason)."""
    resolved: list[ResolvedBet] = []
    errors: list[tuple[ParsedBet, str]] = []

    for pb in parsed:
        match = _find_match(pb.home_team, pb.away_team)
        if not match:
            errors.append((pb, f"no encontré el partido {pb.home_team} v {pb.away_team} en la base"))
            continue

        # If we found it with home/away swapped vs user input, log it
        swapped_note = ""
        if match.get("_swapped"):
            swapped_note = " (orden invertido — el local en la liga es el contrario al pegado)"

        sel, conf, sel_note = _infer_selection(
            pb.market, pb.odds, match["match_id"], pb.selection_hint,
        )

        resolved.append(ResolvedBet(
            match_id=match["match_id"],
            home_team=match["home"],
            away_team=match["away"],
            league=match["league"],
            market=pb.market,
            selection=sel,
            odds=pb.odds,
            stake=pb.stake,
            confidence=conf,
            note=(sel_note + swapped_note).strip(),
        ))
    return resolved, errors


# ---------- Insert ----------

def register_resolved_bets(bets: list[ResolvedBet], mode: Mode = "real") -> RegistrationResult:
    """Insert resolved bets into the picks table with bypass_risk_check.
    Updates bankroll history. Returns counts for the user."""
    from src.tracking.pick_logger import log_pick

    result = RegistrationResult()
    for b in bets:
        # Build a placeholder ValueBet. model_probability=0 + edge=0 signals
        # this is a manual external bet, not a model recommendation.
        vb = ValueBet(
            match_id=b.match_id,
            home_team=b.home_team,
            away_team=b.away_team,
            league=b.league,
            market=b.market,
            selection=b.selection,
            odds=b.odds,
            bookmaker="wplay_external",
            model_probability=0.0,
            fair_odds=0.0,
            edge=0.0,
            confidence=b.confidence,
            recommended_stake=b.stake,
            reasoning="external bet registered via paste",
        )
        try:
            pick_id = log_pick(vb, mode=mode, bypass_risk_check=True)
            result.inserted.append((pick_id, b))
        except Exception as exc:
            logger.exception(f"failed to log external bet for match {b.match_id}")
            result.skipped.append((
                ParsedBet(home_team=b.home_team, away_team=b.away_team, odds=b.odds, stake=b.stake),
                f"insert failed: {exc}",
            ))
    return result
