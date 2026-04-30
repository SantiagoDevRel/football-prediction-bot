# Football Prediction Bot

Sistema de predicción de partidos de fútbol con detección de **value bets** y bankroll management. Combina múltiples modelos estadísticos + features cualitativas generadas por LLM (Claude) para encontrar diferencias entre la probabilidad real estimada y las cuotas de las casas de apuestas.

> **Expectativas reales:** ROI objetivo de **2–8% anual** sobre el monto apostado, sostenido en el largo plazo (1000+ apuestas). No es esquema de plata fácil. El proyecto vale por el aprendizaje técnico + edge sostenible, no por promesas de retornos altos.

---

## Ligas objetivo

- **Premier League** (Inglaterra) — mejor data pública (xG histórico, lineups, etc.)
- **Liga BetPlay Dimayor** (Colombia) — más alineada con Wplay, data más escasa

Vamos a comparar performance del modelo en ambas para entender dónde tiene más edge.

---

## Arquitectura

```
[Datos — TODO free tier]             [Modelos]                       [Output]
 ├─ ESPN public API (live + recent,   ├─ Dixon-Coles (Poisson +       ├─ Detector de value
 │   ambas ligas, sin auth)           │   decay temporal)              │   (cuota_casa vs prob_modelo)
 ├─ football-data.co.uk (CSVs         ├─ Elo dinámico                  ├─ Kelly fraccionado (¼ Kelly)
 │   históricos: results + cuotas     ├─ XGBoost (gradient boosting)   ├─ Telegram bot (push picks)
 │   de cierre, Premier 25+ años)     ├─ Bayesiano (PyMC, da           └─ Dashboard Streamlit
 ├─ Understat (xG histórico Premier)  │   incertidumbre)               
 ├─ Wplay scraper (cuotas, READ-ONLY) ├─ Red neuronal con embeddings
 └─ Noticias / Twitter                └─ LLM (Claude) → features
                                          cualitativas
                                          
                              [Stacking / Ensemble]
                                          │
                                          ▼
                              Probabilidad calibrada por mercado
                              (1X2, O/U 2.5, BTTS, hándicap, tarjetas)
```

### Loop de aprendizaje

Cada apuesta se loggea con TODO el contexto (probas de cada modelo sub, cuota de apertura, cuota de cierre, features del LLM, resultado real, CLV). Semanalmente:

- Re-entrenamiento automático de modelos
- Análisis de errores por liga / mercado / tipo de cuota
- Feature importance tracking
- **CLV (Closing Line Value) es el KPI principal**, no win-rate

---

## Roadmap

| Fase | Entregable | Estado |
|---|---|---|
| **0** | Estructura del repo, schema DB, stubs de módulos | ✅ Listo |
| **1** | ESPN client + Dixon-Coles + Elo + backtest | ✅ Listo (ver `docs/phase1-backtest-report.md`) |
| **2** | LLM features + value detector + pick logger + daily pipeline + manual paper-pick CLI | ✅ Listo |
| **2.5** | Wplay scraper funcional (selectores) + Telegram bot configurado + cron diario | 🟡 Pendiente humano (ver `HANDOFF.md`) |
| **3** | XGBoost + xG (Understat) + Bayesian + ensemble stacking + dashboard | ⏳ Pendiente |

**Gate crítico al final de Fase 1:** si en backtest el modelo no le gana al mercado en CLV, paramos y revisamos diseño. No avanzamos asumiendo que "después se arregla".

---

## Stack

- **Lenguaje:** Python 3.11+
- **Package manager:** `uv` (rápido, moderno)
- **Datos:** ESPN public API (sin auth), football-data.co.uk (CSV histórico gratis), Understat (scrape gratis), Wplay scraper. **Free-tier only, regla dura.**
- **Modelos:** statsmodels, xgboost, pymc, pytorch, scikit-learn
- **LLM:** Anthropic SDK (Claude) para features cualitativas
- **DB:** SQLite (dev local), migración a Postgres si crece
- **Pipeline:** Prefect (Fase 3)
- **Tracking:** MLflow (Fase 3)
- **Notificaciones:** Telegram bot
- **Dashboard:** Streamlit

---

## Modos de operación

1. **Paper trading (default):** el bot predice y simula apuestas con bankroll virtual. Loggea CLV y ROI. Corre 24/7.
2. **Real money:** activado manualmente por el usuario. Muestra picks con stake recomendado, el usuario ejecuta manualmente en Wplay (ver decisión arquitectónica abajo).

### ¿Por qué NO automatizamos la apuesta en Wplay?

- Wplay prohíbe bots en sus T&Cs. Detección → cuenta cerrada + saldo confiscado.
- Riesgo de bug catastrófico (apuesta x10 de stake, repite picks, etc.).
- Credenciales en código = superficie de ataque.

El bot **solo lee cuotas y notifica**. La decisión de apostar y la ejecución las hace el usuario manualmente. Es 30 segundos extra por pick, vale la pena.

---

## Estructura del repo

```
football-prediction-bot/
├── README.md
├── CLAUDE.md                    # Reglas específicas del proyecto para Claude Code
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                     # Pulls crudos de APIs/scrapers
│   ├── processed/               # Datasets limpios para modelos
│   └── db/                      # SQLite local
├── src/
│   ├── config/                  # Settings, env loading
│   ├── data/                    # Fetchers (API-Football, Understat, Wplay)
│   ├── models/                  # Dixon-Coles, Elo, XGBoost, Bayesiano, ensemble
│   ├── betting/                 # Value detector, Kelly, bankroll
│   ├── llm/                     # Cliente Claude para features cualitativas
│   ├── tracking/                # Logging de picks, CLV, métricas
│   └── notifications/           # Telegram bot
├── scripts/                     # Pipelines, jobs, init scripts
├── notebooks/                   # Análisis exploratorio, experimentos
├── tests/                       # Pytest
└── logs/                        # Logs de runs
```

---

## Quick start

> **Nota:** Fase 0 solo deja la estructura armada. Los scrapers y modelos llegan en Fase 1.

```bash
cd football-prediction-bot

# Instalar uv si no está
# curl -LsSf https://astral.sh/uv/install.sh | sh   (Linux/Mac)
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   (Windows)

uv sync                         # Instala dependencias declaradas en pyproject.toml
cp .env.example .env            # Copiar y llenar con tus keys
python scripts/init_db.py       # Crear schema SQLite local
```

---

## Disciplina y reglas duras

1. **Paper trading mínimo 100+ picks con CLV positivo** antes de tocar plata real.
2. **Kelly fraccionado (¼ Kelly)**, nunca full Kelly. La varianza te mata.
3. **Bankroll dedicado** que estés dispuesto a perder. No mezclar con plata de gastos.
4. **Si dos semanas seguidas el CLV es negativo**, pausar plata real, investigar.
5. **No combinadas** (parlays). Apuestas separadas siempre, salvo análisis específico.
6. **Logging obsesivo.** Toda apuesta queda en DB con su contexto completo.
