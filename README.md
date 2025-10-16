# Telegram Smart Money Trading Assistant

## Overview
An asynchronous Telegram bot and FastAPI backend that analyses MT4/MT5 chart screenshots with GPT-4o Vision, logs validated Smart Money zones to SQLite, and monitors live prices to alert traders when price approaches tracked entries.

## Features
- GPT-4o vision analysis using Dami’s IOF Smart Money protocol.
- Retrieval-augmented memory with FAISS-backed embeddings for chart history and uploaded playbooks.
- Document ingestion (TXT/MD/PDF) into the knowledge base and automatic recall during future analyses.
- Optional OpenAI web search augmentation for macro/fundamental queries.
- Voice note transcription and natural language replies; audio alerts (TTS) for price proximity events.
- CSV trade analytics via OpenAI Code Interpreter endpoints.
- Structured JSON validation and SQLite persistence via SQLAlchemy.
- Live price monitoring through Binance or OANDA with APScheduler alerts.
- FastAPI endpoints for zone management and manual alert triggers.
- Test mode for offline development with deterministic responses.

## Setup
1. Clone the repository and create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and populate credentials:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `DATABASE_URL` (default uses SQLite in `data/zones.db`)
   - `PRICE_PROVIDER` (`binance` or `oanda`)
   - Additional provider credentials as required.
4. Initialize the database (optional, runs automatically on start):
   ```bash
   python -m tools.logger
   ```

## Running Locally
- Launch the FastAPI server and Telegram bot together:
  ```bash
  python main.py
  ```
- FastAPI serves on `FASTAPI_HOST:FASTAPI_PORT` (default `http://0.0.0.0:8000`).
- Telegram bot begins polling once the app starts; send charts to receive analyses.

## Environment Options
- `TEST_MODE=true` enables deterministic LLM and price feed responses for offline testing.
- `SCHEDULER_TIMEZONE` sets the APScheduler timezone (default `UTC`).

## FastAPI Endpoints
- `GET /health` – Service heartbeat.
- `GET /zones` – List active zones.
- `GET /zones/{id}` – Retrieve a specific zone.
- `POST /alerts/check` – Trigger immediate price evaluation.

## Telegram Interactions
- Send MT4/MT5 screenshots to receive structured IOF analysis and persistent zone logging.
- Upload strategy notes (`.txt`, `.md`, `.pdf`) to expand the agent’s knowledge base.
- Upload trade journals (`.csv`) to trigger automated stats via Code Interpreter.
- Send voice notes for automatic transcription and conversational responses.
- Ask macro or sentiment questions; the agent invokes web search when keywords such as “macro”, “fundamentals”, or “sentiment” are detected.
- Receive optional MP3 audio alerts whenever price approaches tracked entries.

## Extending
- Add new prompt variations under `prompts/` and update the agent loader.
- Integrate additional price feeds by subclassing `AbstractPriceFeed` in `tools/price_feed.py`.
- Enhance alert logic within `tools/alerts.py` for advanced notification rules.
- Adjust document chunking strategy in `tools/knowledge.py` for bespoke retrieval behaviour.

## Testing
Run automated tests with:
```bash
pytest
```

## Deployment
- Provide Docker runtime variables for secrets; never bake API keys into images.
- Mount persistent volumes for `data/zones.db`, `data/uploads`, `data/memory`, `data/docs`, and `data/audio`.
- Use process supervisors (systemd, Docker, etc.) to keep the FastAPI app active.
- Configure persistent storage for the SQLite database or swap for hosted DB.
