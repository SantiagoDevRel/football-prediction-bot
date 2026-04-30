# Roadmap

Plan vivo. Cada ítem tiene: esfuerzo estimado, por qué importa, criterio de "listo".
Orden = prioridad. Marcamos `[x]` cuando se cierra y `[~]` cuando está en curso.

---

## ✅ DONE

- [x] **Phase 0** — Repo, schema SQLite, módulos
- [x] **Phase 1** — ESPN client, persist layer, Dixon-Coles + Elo, backtest harness
- [x] **Phase 2** — LLM feature extractor, value detector, pick logger, daily pipeline E2E
- [x] **Phase 2.1** — Wplay scraper funcional (1X2 desde league pages)
- [x] **Phase 2.2** — Telegram bot output (one-way) con value bets
- [x] **Phase 2.3** — Cap de edge >30% (filtro de bugs del modelo)
- [x] **Phase 2.4** — Bankroll 100k + mensaje en secciones (cuotas seguras / con riesgo) + más mercados en el modelo (O/U 1.5/3.5, hándicap -1.5)

---

## 🎯 PHASE 2.5 — Usable end-to-end (esta semana)

Objetivo: tener un sistema que vos puedas usar todos los días sin escribir código.

### [ ] Bot interactivo Telegram (`Step B`)
- **Esfuerzo:** 90-120 min
- **Por qué:** evitar comandos en PowerShell. Vos hablás con el bot y ya.
- **Comandos a soportar:**
  - `/start` → menú con botones inline
  - `/picks` → corre pipeline pre-match y manda picks numeradas
  - `/aposte N <plata>` → confirma apuesta del pick #N con stake elegido por vos
  - `/resolver N ganada|perdida` → marca resultado, calcula CLV, actualiza bankroll
  - `/balance` → bankroll actual, picks abiertas, ROI
  - `/historial` → últimas 10 resueltas
- **Listo cuando:** desde el celular podés `/picks`, ver picks, `/aposte 2 5000`, todo loggeado en DB.

### [ ] Wplay scraper multi-mercado
- **Esfuerzo:** 90-120 min
- **Por qué:** hoy solo 1X2 → ~3 picks/día. Con O/U + BTTS + hándicap → 5-10x más oportunidades.
- **Approach:** navegar a cada `/es/e/{event_id}/{slug}` y parsear todos los markets visibles.
- **Mercados a sacar:** 1X2, O/U 1.5/2.5/3.5, BTTS, Hándicap asiático -1.5/+1.5
- **Listo cuando:** el detector de value emite picks en mínimo 4 mercados distintos por día.

### [ ] Auto-resolución de picks vía ESPN
- **Esfuerzo:** 60 min
- **Por qué:** hoy hay que marcar manualmente cada pick como ganada/perdida. Tedioso. ESPN ya nos dice el score final.
- **Approach:** cron diario que: (1) busca picks abiertas con kickoff > 3hrs atrás, (2) consulta ESPN por el score final, (3) resuelve auto, (4) calcula CLV con la última cuota Wplay capturada.
- **Listo cuando:** apostás un viernes, el sábado a la mañana ya tenés el resultado actualizado en el bot sin tocar nada.

### [ ] In-play v0 (Poisson condicional)
- **Esfuerzo:** 60-90 min
- **Por qué:** sin esto, no hay análisis en vivo. V0 no es de clase mundial pero es honesto.
- **Approach:** dado el resultado actual (X-Y al min M), recomputar Dixon-Coles para los minutos restantes con λ y μ escalados por tiempo restante. Útil para Over/Under y BTTS in-play.
- **Disclaimer obligatorio:** "Esto NO es un modelo entrenado en vivo. Es una aproximación matemática. Apostá pesado solo cuando tengamos in-play v1."
- **Listo cuando:** `/envivo` lista partidos in-play + cuotas Wplay + predicción condicional + disclaimer claro.

### [ ] Schedule diario (cron / Windows Task Scheduler)
- **Esfuerzo:** 15 min
- **Por qué:** que el sistema corra solo a las 9 AM cada día.
- **Approach:** `schtasks /create` apuntando a `scripts/run_daily.cmd`.
- **Listo cuando:** tu PC arranca, a las 9 AM te llega el mensaje al Telegram sin que hagas nada.

---

## 🚀 PHASE 3 — Mejor modelo (próximas 2 semanas)

Objetivo: que el modelo le gane al mercado en CLV. Hoy apenas le gana al uniforme baseline.

### [ ] Understat scraper (xG histórico Premier)
- **Esfuerzo:** 2-3 hrs
- **Por qué:** xG es la métrica avanzada por excelencia. Mide calidad de chances, no solo goles. Mejora dramáticamente Premier.
- **Listo cuando:** tenemos xG/xGA por partido para temporadas Premier 2020-25 en DB.

### [ ] XGBoost model con features ingenieradas
- **Esfuerzo:** 4-6 hrs
- **Por qué:** Dixon-Coles solo ve goles e identidad. XGBoost puede ver: rolling form, descanso, head-to-head, xG, viaje, etc.
- **Features iniciales:**
  - Rolling avg goals/xG en 5/10 partidos
  - Rolling form (W/D/L)
  - Days of rest
  - Head-to-head últimos 5
  - Home/away splits
  - LLM flags (one-hot)
- **Listo cuando:** XGBoost le gana a Dixon-Coles en log-loss en backtest 2024/25.

### [ ] News scraper + LLM features wired
- **Esfuerzo:** 3-4 hrs
- **Por qué:** el extractor LLM ya está construido (Haiku 4.5). Falta: traer noticias por partido + alimentárselas + meter las flags como features.
- **Approach:** buscar en Marca/AS/Olé/El Tiempo el día previo al partido por team names → snippets → LLM → flags → XGBoost.
- **Listo cuando:** cada predicción incluye flags del LLM y mejora calibración.

### [ ] Stacking ensemble + calibración
- **Esfuerzo:** 3-4 hrs
- **Por qué:** el promedio simple de DC + Elo + XGBoost no es óptimo. Un meta-modelo aprende cuándo confiar en cada uno.
- **Approach:** stacking con out-of-fold predictions + Platt scaling para calibrar la salida final.
- **Listo cuando:** calibración del ensemble pasa: si decimos 60%, ganan en 60% (±5%) en backtest.

### [ ] Bayesiano (PyMC) opcional
- **Esfuerzo:** 4-6 hrs (incluye instalar PyMC, pesado)
- **Por qué:** da intervalos de confianza. Útil para Kelly: si confianza es baja, apostá menos.
- **Listo cuando:** ensemble incluye un modelo Bayes con su `confidence` populated y eso modula el stake.

### [ ] Backtest robusto + reporte
- **Esfuerzo:** 2-3 hrs
- **Por qué:** validar que cada cambio mejora antes de promoverlo a producción.
- **Listo cuando:** `scripts/backtest.py` corre todos los modelos sobre 2 temporadas y produce reporte comparativo automático.

---

## 🃏 PHASE 4 — Mercados nuevos (después de Phase 3)

Solo arrancar Phase 4 cuando Phase 3 dé un modelo que le gane al mercado en CLV. Antes es construir más arriba sobre cimientos débiles.

### [ ] Tarjetas: scraper + modelo
- **Esfuerzo:** 1-2 días
- **Data source:** SofaScore o FBref (scraping). Necesitamos: cards/match histórico por equipo + árbitro asignado por partido.
- **Modelo:** Poisson sobre tasa promedio de tarjetas de los dos equipos × adjustment por árbitro.
- **Listo cuando:** detectamos value en mercados de tarjetas y backtest muestra CLV positivo en al menos una liga.

### [ ] Córners: scraper + modelo
- **Esfuerzo:** 1-2 días
- **Data source:** mismo que tarjetas (SofaScore o FBref).
- **Modelo:** similar Poisson sobre tasas + ajuste por estilo de juego (xG → más córners).
- **Listo cuando:** lo mismo que tarjetas pero para córners.

### [ ] Mercados de jugadores (anytime scorer, etc.)
- **Esfuerzo:** 2-3 días
- **Data source:** stats por jugador (FBref, Understat per player).
- **Modelo:** probabilidad de gol del jugador dado xG histórico + lineup esperado.
- **Listo cuando:** el bot puede decir "Haaland tiene 65% de marcar en este partido, Wplay paga 2.20 a anytime, edge +43% (cap a 30% lo filtra)".

---

## 🔴 PHASE 5 — In-play real

### [ ] In-play model v1 entrenado en datos minuto-a-minuto
- **Esfuerzo:** 2-3 días
- **Data source:** SofaScore live commentary, FotMob historical, o StatsBomb open data
- **Modelo:** opciones:
  - LSTM sobre secuencia de eventos del partido
  - Gradient boosting sobre (score, minute, red cards, shots in last 10', possession, xG so far)
- **Listo cuando:** in-play v1 le gana a in-play v0 en CLV en backtest sobre partidos 2024/25.

### [ ] Multi-bookie comparator
- **Esfuerzo:** 1-2 días
- **Por qué:** comparar Wplay con Bet365, Pinnacle, Betsson. La cuota más alta entre todas = mejor CLV.
- **Listo cuando:** el detector de value compara cuotas de 3+ casas y siempre recomienda apostar en la mejor.

---

## 🛡️ PHASE 6 — Producción / plata real

Solo arrancar cuando: 100+ paper picks con CLV positivo sostenido + Phase 3 completo.

### [ ] Tests automatizados (unit + integration)
- **Esfuerzo:** 4-6 hrs
- **Cobertura objetivo:** Kelly, value detector, persist, pipeline E2E con HTTP mocks.

### [ ] Re-entrenamiento automático semanal
- **Esfuerzo:** 2-3 hrs
- **Approach:** GitHub Actions o cron local. Re-entrena, corre backtest, no promueve si performance bajó.

### [ ] Drift detection
- **Esfuerzo:** 2 hrs
- **Approach:** Evidently sobre feature distributions. Alerta si la data cambia sustancialmente vs entrenamiento.

### [ ] Risk management adicional
- **Esfuerzo:** 2-3 hrs
- **Items:**
  - Cap total de exposición diaria (ej: max 20% del bankroll en un día)
  - Stop-loss: pausar real-money si el bankroll cae 25% en una racha
  - Cooldown después de N losses seguidas

### [ ] Streamlit dashboard
- **Esfuerzo:** 4-6 hrs
- **Páginas:**
  - Bankroll over time
  - ROI / CLV rolling
  - Calibración (predicted vs realized)
  - Pick history filtrable
  - Open positions

---

## 📋 Cómo trackeamos esto

- Este archivo es la fuente de verdad. Lo committeamos en `docs/ROADMAP.md`.
- Cada PR/commit que cierre un ítem actualiza el `[ ]` a `[x]` en el mismo commit.
- En el chat de Claude usamos el sistema de TaskCreate para los próximos 5-10 ítems activos.
- Si un ítem cambia de scope, se documenta el cambio acá con fecha.

## 📅 Mi recomendación de orden

1. **Esta semana** — todo Phase 2.5. Eso te da el sistema usable.
2. **Próxima semana** — Phase 3 ítems 1-3 (Understat + XGBoost + LLM features). Acá viene la mejora de modelo de verdad.
3. **Semana 3** — Phase 3 ítems 4-5 (stacking + backtest robusto) para validar que el modelo realmente le gana al mercado.
4. **Semana 4+** — paper trading con foco. Acumulamos 100+ picks. Si CLV es positivo, recién ahí Phase 4 (tarjetas/córners) y Phase 6 (real money).
5. **Phase 5** (in-play real) y **Phase 6** completo se postergan hasta tener edge demostrado en pre-match. Sin eso, son features sobre cimientos débiles.
