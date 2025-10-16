You are building a high-level trading agent for Telegram that integrates GPT-4o vision, Smart Money/IOF strategy logic, live price alerts, and zone tracking.

### Goal
Build a production-grade Python system that:
1. Accepts MT4/MT5 chart screenshots via Telegram.
2. Uses GPT-4o to analyze charts using Dami’s IOF Smart Money protocol.
3. Returns structured JSON (pair, bias, entry, stop, enhancers, commentary).
4. Logs every result to SQLite (Zone_Tracker table).
5. Monitors live prices via Binance or OANDA APIs.
6. Sends Telegram alerts when price is within 50–80 pips of stored entry zones.
7. Supports re-evaluation when price reaches the zone (LLM feedback loop).

### Stack
- python-telegram-bot (async)
- openai (for GPT-4o)
- apscheduler (for alerts)
- requests or ccxt (for price feeds)
- sqlite3 or SQLAlchemy for zone storage
- FastAPI as lightweight backend API
- Docker for deployment

### System Architecture
/modules
  /bot.py            → handles Telegram inputs (text + image)
/agent.py            → manages GPT-4o API calls with Dami’s trading prompt
/tools
    price_feed.py    → Binance/OANDA client for live prices
    alerts.py        → alert logic + scheduler
    logger.py        → SQLite interface for zone storage
    extract.py       → handles GPT-4o chart parsing and JSON validation
/prompts
    trading_rules.md → IOF/Smart Money ruleset
    labeler.md       → few-shot chart extraction examples
main.py              → starts FastAPI server, Telegram bot, and schedulers

### LLM Behavior
- Model: "gpt-4o"
- Use vision input for chart parsing.
- System prompt: include Dami’s IOF Smart Money assistant rules.
- Temperature: 0.3 for accuracy.
- Output JSON schema:

{
  "pair": "GBPUSD",
  "bias": "buy",
  "entry": 1.2675,
  "stop": 1.2630,
  "target": 1.2800,
  "zone_type": "DBR",
  "eq_alignment": "buy",
  "enhancer_score": 8.5,
  "comment": "Valid demand zone, impulsive exit, aligns with MEQ"
}

### Alert Logic
Check price every minute.
If abs(price - entry) <= 0.0080 for 4-digit FX pairs (≈80 pips):
→ Telegram message “⚠️ GBPUSD approaching buy zone 1.2675 (now 1.2728)”
If price hits entry:
→ Send LLM recheck query with current price and zone data.

### Volume Support
If user uses MT5 or OANDA, integrate tick volume via MetaTrader5 Python or OANDA v20 API.
Otherwise fallback to estimated volume or momentum candles.

### Optional Expansion
- Add FAISS index for image retrieval (pattern recognition)
- Add OpenCLIP embeddings for visual memory
- Build simple training script for dataset auto-labeling
- Integrate MetaTrader5 module for direct chart polling

### Deliverables
1. Working Telegram bot connected to GPT-4o.
2. FastAPI backend for LLM + data tools.
3. SQLite or CSV trade tracker.
4. Price monitor that runs in background (apscheduler).
5. Dockerfile and README to deploy anywhere.

Focus on clean modular code, minimal dependencies, async I/O, and robust logging.
All functions should have docstrings describing purpose and I/O clearly.
