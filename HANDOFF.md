# Handoff

**Repo:** https://github.com/SantiagoDevRel/football-prediction-bot
**Estado:** Phase 2.5 completa. El sistema corre solo si activás los schedules.

---

## Cómo se usa el sistema (UX completa)

Todo se hace desde Telegram. El bot escucha 24/7 (si está activo el `FootballBotPersist` task). Comandos:

| Comando | Qué hace |
|---|---|
| `/start` | Menú principal con botones |
| `/picks` | Analiza partidos próximos 2 días en Premier + Liga BetPlay, te lista picks numeradas |
| `/envivo` | Lista partidos en vivo + predicciones in-play v0 (con disclaimer claro) |
| `/analizar Arsenal Fulham` | Análisis detallado de UN partido (todos los mercados) |
| `/aposte 3 5000` | Registra apuesta del pick #3 con stake $5000 (sin stake = usa el sugerido) |
| `/resolver_auto` | Resuelve manualmente cualquier pick cuya partido haya terminado |
| `/resolver 42 ganada` | Manual override (si auto-resolver falla por alguna razón) |
| `/balance` | Bankroll, picks abiertas, ROI rolling 30 días |
| `/historial` | Últimas 10 picks resueltas con P&L y CLV |

### Lenguaje natural (Claude NLU)

Todo mensaje que NO empieza con `/` lo interpreta Claude Haiku 4.5 vía tool-use
estructurado y lo despacha al comando correcto con filtros aplicados. Ejemplos:

| Lo que escribes | Lo que ejecuta |
|---|---|
| "dame el top pick de hoy en betplay" | `/picks` filtrado por liga_betplay + today + top_only |
| "qué hay en vivo" | `/envivo` |
| "analiza nacional vs millonarios" | `/analizar Nacional Millonarios` |
| "cómo voy de balance" | `/balance` |
| "aposté el 3 con 5000" | `/aposte 3 5000` |
| "dame algo de champions este finde" | `/picks` con filtro champions_league + weekend (warning si liga no activa) |
| "hola, qué tal" | smalltalk con respuesta corta en español colombiano |

Costo: ~$0.001 por mensaje (Haiku 4.5). Necesita `ANTHROPIC_API_KEY` en `.env`.
Sin key, los mensajes de texto libre devuelven una nota pidiendo usar comandos slash.

---

## Activar el sistema en Windows (corre solo)

Una vez para programar todo (ejecutar como Administrador):

```cmd
cd C:\Users\STZTR\Desktop\claude-code-environment\football-prediction-bot
scripts\install_schedule.cmd
```

Esto crea 3 tareas programadas:

1. **FootballBotDaily** — todos los días 9 AM corre `daily_pipeline.py` → te llega resumen al Telegram
2. **FootballBotResolve** — cada hora resuelve picks finalizadas → te llega notificación si hubo
3. **FootballBotPersist** — cada 15 min verifica que el bot listener esté vivo, lo reinicia si murió

Verificar:
```cmd
schtasks /query /tn FootballBotDaily
schtasks /query /tn FootballBotResolve
schtasks /query /tn FootballBotPersist
```

Logs:
- `logs/daily_YYYYMMDD.log`
- `logs/resolve_YYYYMMDD.log`
- `logs/wplay_debug/` (raw HTML de Wplay para debug si scraper falla)

Desinstalar:
```cmd
schtasks /delete /tn FootballBotDaily /f
schtasks /delete /tn FootballBotResolve /f
schtasks /delete /tn FootballBotPersist /f
```

---

## Si querés correr cosas a mano

```powershell
cd C:\Users\STZTR\Desktop\claude-code-environment\football-prediction-bot
$env:PYTHONUTF8 = "1"

# Pipeline diario manual
.venv\Scripts\python.exe scripts\daily_pipeline.py

# Auto-resolver manual (también lo hace /resolver_auto en el bot)
.venv\Scripts\python.exe scripts\resolve_picks.py

# Re-entrenar modelos
.venv\Scripts\python.exe scripts\train_models.py

# Re-correr backtest
.venv\Scripts\python.exe scripts\backtest.py --league premier_league

# Bot listener (foreground)
.venv\Scripts\python.exe scripts\telegram_bot.py
```

---

## Arquitectura

```
[ESPN public API]                      [Models]
 ├─ fetch_scoreboard()                  ├─ Dixon-Coles (Poisson + decay temporal)
 ├─ fetch_season_history()              ├─ Elo dinámico
 └─ status: scheduled/live/finished     └─ ensemble simple promedio (Phase 3 → stacking)

[Wplay scraper]
 ├─ scrape_league()             1X2 desde league pages (rápido, ~10s)
 ├─ scrape_match_markets()      todos los mercados desde match page (~3s)
 └─ scrape_all_with_markets()   E2E para daily pipeline (~60s)

[In-play v0]
 └─ condition_on_state(pre_match, hg, ag, minute)
    Recompute Poisson sobre minutos restantes

[LLM Feature Extractor]
 └─ Anthropic Claude Haiku 4.5 + prompt cache + vocabulario controlado
    (cableo a XGBoost en Phase 3)

[Pipeline]
 ├─ run_pipeline_core()         Predicción + Wplay + value detection (NO logs)
 ├─ daily_pipeline.main()       cron entry: auto-resolve + run + log + Telegram
 └─ /picks (bot)                stages picks numeradas, vos /aposte N stake

[DB SQLite]
 11 tablas: leagues, teams, matches, odds_snapshots, predictions,
 qualitative_features, picks, bankroll_history, model_performance,
 staged_picks (bot), config
```

---

## Estado de cada fase

✅ **Phase 0** — Estructura, schema, módulos
✅ **Phase 1** — ESPN client, Dixon-Coles + Elo, backtest
✅ **Phase 2** — LLM extractor, value detector, pick logger, daily pipeline
✅ **Phase 2.5 COMPLETA**:
  - Bot interactivo (`/picks`, `/aposte`, `/balance`, `/historial`)
  - `/analizar <home> <away>` con multi-mercado
  - Wplay scraper multi-mercado (1X2 + BTTS + O/U 1.5/2.5/3.5)
  - Auto-resolución vía ESPN (cron horario o `/resolver_auto`)
  - In-play v0 + `/envivo`
  - Schedule diario Windows Task Scheduler

🟡 **Phase 3 — pendiente:**
- Understat xG scraper
- XGBoost con features ingenieradas
- News scraper + LLM features wired al ensemble
- Stacking ensemble + calibración (Platt scaling)
- Backtest robusto con CLV cuando lleguen cuotas históricas

⏳ **Phase 4** — Tarjetas, córners, jugadores. Solo después de validar edge en Phase 3.
⏳ **Phase 5** — In-play v1 real, multi-bookie.
⏳ **Phase 6** — Tests, re-train automático, risk mgmt, dashboard. Pre-requisito para plata real.

---

## Reglas de seguridad y disciplina

- **Mode default: paper.** No tocar plata real hasta 100+ picks resueltos con CLV positivo sostenido.
- **Filtros automáticos:** edge entre 5% y 30% (arriba de 30% casi siempre es bug del modelo).
- **Stake:** ¼ Kelly fraccional con cap 5% del bankroll por apuesta.
- **Wplay es READ-ONLY.** Nunca automatizamos apuestas allí. Sus T&Cs prohíben bots → cierre de cuenta + saldo confiscado.
- **API keys.** Solo en `.env` (gitignored). Si una key toca chat / log / Slack → rotar inmediatamente.
- **No combinadas.** Apuestas separadas siempre.

---

## Métricas que mirar

- **CLV (Closing Line Value):** la única prueba real de edge. Hoy no se mide (no capturamos closing odds aún) — Phase 3 add.
- **Win rate ≠ ROI.** Podés ganar 60% de las apuestas y perder plata si las cuotas son malas.
- **Brier score y log-loss.** Calibración del modelo. Phase 1 backtest dio Brier 0.61 en BetPlay vs uniforme 0.67 — hay señal.
- **ROI rolling 30d.** Aparece en `/balance`.

---

*Última actualización: 2026-04-30. Update con `update HANDOFF` en chat.*
