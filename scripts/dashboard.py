"""Streamlit dashboard for tracking your paper-trading performance.

Run:
    streamlit run scripts/dashboard.py

Pages:
    - Overview: bankroll over time + risk summary
    - Picks: open picks + history with P&L and CLV
    - Calibration: predicted probability vs realized rate
    - Models: per-model performance from backtest reports
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.betting.risk_manager import risk_summary  # noqa: E402
from src.tracking.pick_logger import compute_rolling_metrics  # noqa: E402


st.set_page_config(page_title="Football Prediction Bot", layout="wide", page_icon="⚽")
st.title("⚽ Football Prediction Bot — Dashboard")

mode = st.sidebar.radio("Modo", ["paper", "real"], index=0)


# ---------- Helpers ----------

@st.cache_data(ttl=30)
def load_picks(mode: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(settings.db_path))
    df = pd.read_sql_query(
        """
        SELECT p.id, p.market, p.selection, p.odds_taken, p.stake, p.placed_at,
               p.won, p.payout, p.clv, p.resolved_at, p.edge, p.model_probability,
               h.name AS home, a.name AS away, l.name AS league
          FROM picks p
          JOIN matches m ON p.match_id = m.id
          LEFT JOIN teams h ON m.home_team_id = h.id
          LEFT JOIN teams a ON m.away_team_id = a.id
          LEFT JOIN leagues l ON m.league_id = l.id
         WHERE p.mode = ?
         ORDER BY p.placed_at DESC
        """,
        conn, params=(mode,),
    )
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_bankroll_history(mode: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(settings.db_path))
    df = pd.read_sql_query(
        "SELECT id, delta, balance, created_at, note FROM bankroll_history WHERE mode = ? ORDER BY id ASC",
        conn, params=(mode,),
    )
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


# ---------- Sections ----------

picks = load_picks(mode)
bankroll_df = load_bankroll_history(mode)
risk = risk_summary(mode)
metrics = compute_rolling_metrics(mode, days=30)

st.header("📊 Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bankroll", f"${risk['bankroll']:,.0f}", f"de ${risk['peak']:,.0f} peak")
c2.metric("Drawdown", f"{risk['drawdown_pct']*100:.1f}%",
          delta_color=("inverse" if risk["drawdown_pct"] > 0.05 else "normal"))
c3.metric("Picks resueltas (30d)", f"{metrics['n']}", f"WR {metrics['win_rate']:.0%}")
c4.metric("ROI 30d", f"{metrics['roi']*100:+.1f}%", f"P&L ${metrics['total_pnl']:+,.0f}")

if risk["stop_loss_active"]:
    st.error("🔴 STOP-LOSS ACTIVO. Pausá nuevas apuestas hasta entender el drawdown.")
elif risk["consecutive_losses"] >= 4:
    st.warning(f"⚠️ {risk['consecutive_losses']} pérdidas seguidas. Cooldown a partir de {risk['cooldown_threshold']}.")

# Bankroll curve
st.subheader("Curva de bankroll")
if not bankroll_df.empty:
    fig = px.line(bankroll_df, x="created_at", y="balance", markers=True,
                  labels={"created_at": "Fecha", "balance": "Bankroll (COP)"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Todavía no hay historial de bankroll. Empezá a apostar para ver la curva.")

# Picks tables
st.header("📋 Picks")

open_picks = picks[picks["won"].isnull()]
resolved = picks[picks["won"].notnull()].copy()
resolved["pnl"] = resolved["payout"].fillna(0) - resolved["stake"].fillna(0)

t1, t2, t3 = st.tabs([f"Abiertas ({len(open_picks)})",
                       f"Resueltas ({len(resolved)})",
                       "Calibración"])
with t1:
    if open_picks.empty:
        st.info("No hay picks abiertas.")
    else:
        st.dataframe(
            open_picks[["id", "league", "home", "away", "market", "selection",
                        "odds_taken", "stake", "edge", "placed_at"]],
            use_container_width=True,
        )
with t2:
    if resolved.empty:
        st.info("No hay picks resueltas todavía.")
    else:
        st.dataframe(
            resolved[["id", "league", "home", "away", "market", "selection",
                      "odds_taken", "stake", "won", "pnl", "clv", "resolved_at"]],
            use_container_width=True,
        )
        # Summary by market
        by_market = (
            resolved.groupby("market")
            .agg(n=("id", "count"), wr=("won", "mean"),
                 pnl=("pnl", "sum"), avg_clv=("clv", "mean"))
            .round(3)
        )
        st.subheader("Performance por mercado")
        st.dataframe(by_market, use_container_width=True)

with t3:
    st.write(
        "**Calibración** = ¿cuando el modelo dice X%, gana en X% de las veces?  \n"
        "Una recta perfecta significa modelo bien calibrado."
    )
    if resolved.empty or len(resolved) < 10:
        st.info("Necesito al menos 10 picks resueltas para mostrar calibración.")
    else:
        # Bin by predicted probability
        r = resolved[resolved["model_probability"].notnull()].copy()
        r["prob_bin"] = pd.cut(
            r["model_probability"],
            bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            labels=["<20%", "20-30%", "30-40%", "40-50%",
                    "50-60%", "60-70%", "70-80%", ">80%"],
        )
        cal = r.groupby("prob_bin").agg(
            n=("id", "count"),
            avg_pred=("model_probability", "mean"),
            realized=("won", "mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="gray"),
                                 name="Perfectamente calibrado"))
        fig.add_trace(go.Scatter(x=cal["avg_pred"], y=cal["realized"],
                                 mode="markers+lines+text",
                                 text=cal["n"].astype(str),
                                 textposition="top center",
                                 name="Modelo (n por punto)"))
        fig.update_layout(
            xaxis=dict(range=[0, 1], title="Probabilidad predicha"),
            yaxis=dict(range=[0, 1], title="Frecuencia realizada"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cal, use_container_width=True)

# Backtest reports
st.header("🧪 Reportes de backtest")
reports = sorted((ROOT / "docs").glob("backtest-*.md"), reverse=True)
if not reports:
    st.info("No hay reportes de backtest todavía. Corré scripts/backtest_full.py.")
else:
    selected = st.selectbox("Reporte", [r.name for r in reports])
    st.markdown((ROOT / "docs" / selected).read_text(encoding="utf-8"))

st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
           "Datos en `data/db/football_bot.db`. "
           "Para refrescar: corré `/picks` o `daily_pipeline.py`.")
