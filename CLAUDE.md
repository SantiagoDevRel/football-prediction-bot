# CLAUDE.md — Football Prediction Bot

Reglas específicas del proyecto. Las globales (`~/.claude/CLAUDE.md`) aplican también — esto agrega contexto del dominio y decisiones arquitectónicas para que Claude no las re-cuestione cada sesión.

---

## Qué es esto

Bot que detecta value bets en fútbol combinando modelos estadísticos (Dixon-Coles, Elo, XGBoost, Bayesiano) + features cualitativas generadas por LLM. Apunta a Premier League y Liga BetPlay como ligas objetivo iniciales.

**KPI principal: CLV (Closing Line Value), no win rate.** Si nuestras cuotas promedio son mejores que las de cierre del mercado, tenemos edge real, aunque ganemos/perdamos en una semana.

---

## Decisiones arquitectónicas (no re-cuestionar sin razón fuerte)

### 0. Free-tier only (regla dura, alineada con la-polla)

- Toda fuente de datos en plan gratuito. Antes de proponer un servicio pago, parate y proponé el tradeoff.
- Stack actual: ESPN public (sin auth), football-data.co.uk (CSV público), Understat (scrape), Wplay scraper.
- Si una limitación bloquea un feature → listar tradeoff y dejar que el usuario decida si paga.

### 1. Wplay es READ-ONLY
- Solo scrapeamos cuotas, **NUNCA automatizamos apuestas**.
- Razón: T&Cs de Wplay prohíben bots → cierre de cuenta + saldo confiscado.
- El usuario ejecuta apuestas manualmente. El bot solo notifica.
- Si en algún momento se considera automatizar, requiere conversación explícita y migrar a casa que tolere bots (Pinnacle, Betfair Exchange).

### 2. Paper trading antes de plata real
- Mínimo 100+ picks con CLV positivo antes de activar real money.
- El usuario activa explícitamente el modo real, default es paper.

### 3. SQLite, no Postgres (por ahora)
- Dev local, simple, cero infra.
- Migrar a Postgres cuando: (a) corra en VPS, o (b) volumen de datos > 1GB.

### 4. Kelly fraccionado, nunca full
- ¼ Kelly por default. La varianza con full Kelly arruina bankrolls aunque el modelo tenga edge real.

### 5. No combinadas / parlays
- Apuestas separadas. Multiplicar cuotas multiplica varianza más rápido que el edge.

### 6. Modelos en ensemble, no uno solo
- Cada modelo tiene fortalezas distintas. El stacking aprende cuándo confiar en cada uno.
- No descartar modelos viejos cuando llegue uno nuevo — sumarlos al ensemble.

### 7. LLM para features cualitativas, NO para predicciones numéricas directas
- Claude lee noticias / lineups / contexto y genera flags como `["palmeiras_rota_titulares", "cerro_juega_altura"]`.
- Esos flags son features del modelo numérico, no la predicción final.
- **Nunca pedirle a Claude "dame la probabilidad de que gane X"** — está mal calibrado para eso.

---

## Reglas de código

### Python
- Python 3.11+
- Type hints obligatorios en funciones públicas
- `uv` para gestión de dependencias (no pip directo, no poetry)
- Formateador: `ruff format`
- Linter: `ruff check`
- Sin imports relativos largos: usar `from src.models import ...` (paquete absoluto)

### Datos
- Raw data → `data/raw/` (immutable, nunca se modifica una vez bajado)
- Datos procesados → `data/processed/`
- DB local → `data/db/football_bot.db` (SQLite, .gitignored)
- **Nunca commitear data real**, solo schemas y samples sintéticos

### Secretos
- TODO secreto en `.env`, nunca hardcodeado
- `.env` está gitignored, `.env.example` se commitea con keys vacías
- API keys, tokens de Telegram, etc. → solo en `.env`
- Si una key se pega en chat / logs / Slack → rotar inmediatamente

### Tests
- Pytest. Cada módulo de `src/` tiene su `tests/test_<modulo>.py`.
- Modelos: tests con datos sintéticos + assertion sobre métricas mínimas en backtest.
- Scrapers: tests con HTML/JSON fixtures, no llamadas reales a la API en CI.

### Logs
- `logs/` está gitignored
- Usar `loguru` (más amigable que stdlib logging)
- Nunca loggear: API keys, credenciales, body completos de requests con tokens

---

## Workflow de trabajo

### Antes de implementar un modelo nuevo
1. Justificar por qué (qué hueco cubre que los actuales no)
2. Definir métrica de éxito (CLV, log-loss, calibración)
3. Backtest sobre 2+ temporadas antes de considerar producción

### Antes de mover paper → real
1. 100+ picks logueados
2. CLV promedio positivo
3. Calibración decente (probabilidades estimadas se acercan a frecuencias reales)
4. Confirmación explícita del usuario

### Cuando un modelo falla
- Loggear contexto completo del fallo
- No "parchar" con reglas hardcoded sin entender la causa
- Si dos intentos no resuelven, pausar y reportar antes del tercer intento (regla global)

---

## Qué NO hacer

- ❌ Pedirle a Claude predicciones numéricas directas ("¿qué probabilidad tiene X de ganar?")
- ❌ Automatizar apuestas en Wplay u otras casas con T&Cs anti-bot
- ❌ Usar full Kelly
- ❌ Apuestas combinadas
- ❌ Saltarse paper trading
- ❌ Commitear data real, .env, o cualquier API key
- ❌ Predecir mercados sin data de calidad (Liga 2 colombiana sin xG, etc.) — ser claro sobre limitaciones

---

## Glosario rápido

- **CLV (Closing Line Value):** diferencia entre cuota a la que apostamos vs cuota de cierre. Si fue mejor, tenemos edge real.
- **xG / xGA:** expected goals / expected goals against. Métrica avanzada de calidad de chances.
- **Kelly Criterion:** fórmula para tamaño óptimo de apuesta dado un edge. `f = (bp - q) / b`. Fraccionado = multiplicar por 0.25 para reducir varianza.
- **Dixon-Coles:** modelo Poisson modificado con corrección para resultados bajos y decay temporal de pesos.
- **Elo dinámico:** rating que se actualiza después de cada partido. Más responsivo que ratings fijos.
- **Stacking:** ensemble donde un meta-modelo aprende cómo pesar predicciones de modelos base.
- **Calibración:** que las probabilidades estimadas correspondan a frecuencias reales (si decimos 60%, deberían ganar el 60% de las veces a la larga).
