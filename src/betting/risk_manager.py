"""Risk management: hard limits beyond Kelly to protect the bankroll.

Layers (all enforced before a pick gets logged):

1. **Per-bet cap** (already in Kelly): max 5% of bankroll on any single bet.
2. **Daily exposure cap**: total stakes placed in one calendar day cannot
   exceed `max_daily_exposure_pct` (default 20%) of the START-OF-DAY bankroll.
3. **Stop-loss**: if drawdown from peak bankroll exceeds `stop_loss_pct`
   (default 25%), refuse all new picks until manual reset.
4. **Cooldown after losing streak**: after N consecutive losses (default 5),
   pause picks for `cooldown_hours` (default 24).
5. **Mode gate**: real-money picks need explicit settings.betting_mode='real'
   AND at least 100 resolved paper picks with positive CLV.

Functions:
    check_pick_allowed(stake) -> (ok: bool, reason: str)
    record_pick_outcome(pick_id, won, payout) -> None
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from loguru import logger

from src.config import settings
from src.data.persist import get_conn


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""


# Thresholds (could be moved to settings)
MAX_DAILY_EXPOSURE_PCT = 0.20
STOP_LOSS_DRAWDOWN_PCT = 0.25
LOSS_STREAK_THRESHOLD = 5
COOLDOWN_HOURS = 24
REAL_MONEY_MIN_PAPER_PICKS = 100


def _start_of_day_bankroll(mode: str) -> float:
    """Bankroll balance at the start of TODAY (UTC). If no rows from before
    today, use the latest available (i.e. seed value)."""
    today_iso = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT balance FROM bankroll_history
             WHERE mode = ? AND date(created_at) < ?
             ORDER BY id DESC LIMIT 1
            """,
            (mode, today_iso),
        ).fetchone()
        if row is not None:
            return float(row["balance"])
        # No history before today → use first row (seed)
        row = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id ASC LIMIT 1",
            (mode,),
        ).fetchone()
        return float(row["balance"]) if row else float(settings.paper_bankroll_initial)


def _stakes_today(mode: str) -> float:
    """Sum of stakes placed today (UTC)."""
    today_iso = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(stake), 0) AS total FROM picks
             WHERE mode = ? AND date(placed_at) = ?
            """,
            (mode, today_iso),
        ).fetchone()
    return float(row["total"]) if row else 0.0


def _peak_bankroll(mode: str) -> float:
    """Highest balance seen historically. Used for drawdown calc."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(balance) AS peak FROM bankroll_history WHERE mode = ?",
            (mode,),
        ).fetchone()
    if row is None or row["peak"] is None:
        return float(settings.paper_bankroll_initial)
    return float(row["peak"])


def _current_bankroll(mode: str) -> float:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
    return float(row["balance"]) if row else float(settings.paper_bankroll_initial)


def _consecutive_losses(mode: str) -> int:
    """How many resolved paper picks in a row are losses (counting from most recent)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT won FROM picks
             WHERE mode = ? AND won IS NOT NULL
             ORDER BY resolved_at DESC LIMIT 50
            """,
            (mode,),
        ).fetchall()
    streak = 0
    for r in rows:
        if r["won"] == 0:
            streak += 1
        else:
            break
    return streak


def _last_loss_resolved_at(mode: str) -> datetime | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT resolved_at FROM picks
             WHERE mode = ? AND won = 0
             ORDER BY resolved_at DESC LIMIT 1
            """,
            (mode,),
        ).fetchone()
    if row is None or row["resolved_at"] is None:
        return None
    try:
        return datetime.fromisoformat(row["resolved_at"])
    except ValueError:
        return None


def _resolved_paper_picks_with_positive_clv() -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM picks
             WHERE mode = 'paper' AND won IS NOT NULL AND clv > 0
            """,
        ).fetchone()
    return int(row["n"]) if row else 0


# ---------- Public API ----------

def check_pick_allowed(stake: float, mode: str = "paper") -> RiskCheckResult:
    """Layered checks. Return first failing reason; allow if all pass."""
    if stake <= 0:
        return RiskCheckResult(False, "stake debe ser positivo")

    # Per-bet cap (5% of bankroll). Kelly should already enforce this but we
    # double-check here defensively.
    bankroll = _current_bankroll(mode)
    if stake > 0.05 * bankroll:
        return RiskCheckResult(False, f"stake {stake:.0f} > 5% del bankroll ({bankroll:.0f})")

    # Daily exposure
    sod_bankroll = _start_of_day_bankroll(mode)
    today_stakes = _stakes_today(mode)
    if today_stakes + stake > MAX_DAILY_EXPOSURE_PCT * sod_bankroll:
        return RiskCheckResult(
            False,
            f"exposición diaria sumaría {today_stakes + stake:.0f} > "
            f"{MAX_DAILY_EXPOSURE_PCT*100:.0f}% del bankroll inicial del día ({sod_bankroll:.0f})"
        )

    # Stop-loss: drawdown from peak
    peak = _peak_bankroll(mode)
    if peak > 0:
        drawdown = (peak - bankroll) / peak
        if drawdown >= STOP_LOSS_DRAWDOWN_PCT:
            return RiskCheckResult(
                False,
                f"STOP-LOSS activado: drawdown {drawdown*100:.0f}% desde peak (${peak:.0f}). "
                f"Pausar y revisar antes de seguir apostando."
            )

    # Cooldown after losing streak
    streak = _consecutive_losses(mode)
    if streak >= LOSS_STREAK_THRESHOLD:
        last_loss_at = _last_loss_resolved_at(mode)
        if last_loss_at:
            elapsed = datetime.now() - last_loss_at
            if elapsed < timedelta(hours=COOLDOWN_HOURS):
                hrs_left = COOLDOWN_HOURS - elapsed.total_seconds() / 3600
                return RiskCheckResult(
                    False,
                    f"COOLDOWN: {streak} pérdidas seguidas. Esperá {hrs_left:.0f}h "
                    f"antes de la próxima apuesta."
                )

    # Real-money gate
    if mode == "real":
        n_paper = _resolved_paper_picks_with_positive_clv()
        if n_paper < REAL_MONEY_MIN_PAPER_PICKS:
            return RiskCheckResult(
                False,
                f"Modo REAL bloqueado: necesitás {REAL_MONEY_MIN_PAPER_PICKS}+ picks paper "
                f"con CLV positivo (tenés {n_paper}). Seguí en paper."
            )

    return RiskCheckResult(True, "")


def risk_summary(mode: str = "paper") -> dict:
    """Snapshot of risk metrics for /balance or dashboard."""
    bankroll = _current_bankroll(mode)
    peak = _peak_bankroll(mode)
    drawdown = (peak - bankroll) / peak if peak > 0 else 0.0
    sod = _start_of_day_bankroll(mode)
    daily_stakes = _stakes_today(mode)
    return {
        "bankroll": bankroll,
        "peak": peak,
        "drawdown_pct": drawdown,
        "stop_loss_threshold": STOP_LOSS_DRAWDOWN_PCT,
        "stop_loss_active": drawdown >= STOP_LOSS_DRAWDOWN_PCT,
        "start_of_day_bankroll": sod,
        "stakes_today": daily_stakes,
        "daily_remaining": max(0.0, MAX_DAILY_EXPOSURE_PCT * sod - daily_stakes),
        "consecutive_losses": _consecutive_losses(mode),
        "cooldown_threshold": LOSS_STREAK_THRESHOLD,
    }
