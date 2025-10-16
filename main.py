"""Application entrypoint launching FastAPI, Telegram bot, and schedulers."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config import settings
from modules.agent import TradingAgent
from modules.bot import TelegramBot
from tools.alerts import PriceAlertService
from tools.logger import get_active_zones, get_zone, init_db
from tools.price_feed import AbstractPriceFeed, close_price_feed, get_price_feed


def configure_logging() -> None:
    """Configure root logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class AppState:
    """Container for long-lived services."""

    def __init__(self) -> None:
        self.agent: Optional[TradingAgent] = None
        self.price_feed: Optional[AbstractPriceFeed] = None
        self.bot: Optional[TelegramBot] = None
        self.alerts: Optional[PriceAlertService] = None
        self.bot_task: Optional[asyncio.Task[None]] = None


STATE = AppState()


def uploads_path() -> Path:
    path = Path("data/uploads")
    path.mkdir(parents=True, exist_ok=True)
    return path


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logging.getLogger(__name__).info("Starting application")
    await init_db()

    STATE.agent = TradingAgent(Path("prompts"))
    STATE.price_feed = await get_price_feed()
    STATE.bot = TelegramBot(STATE.agent, uploads_path())
    application = STATE.bot.build()
    STATE.alerts = PriceAlertService(STATE.price_feed, STATE.agent, application.bot)
    STATE.bot.attach_alert_service(STATE.alerts)
    STATE.alerts.start()

    await application.initialize()
    await application.start()
    if application.updater:
        await application.updater.start_polling()

    try:
        yield
    finally:
        logging.getLogger(__name__).info("Shutting down application")
        if STATE.alerts:
            await STATE.alerts.shutdown()
        if application.updater:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
        if STATE.agent:
            await STATE.agent.close()
        if STATE.price_feed:
            await close_price_feed(STATE.price_feed)


app = FastAPI(title="Trading Assistant", lifespan=lifespan)


async def active_alerts() -> PriceAlertService:
    if not STATE.alerts:
        raise HTTPException(status_code=503, detail="Alert service unavailable")
    return STATE.alerts


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/zones")
async def list_zones() -> List[Dict]:
    zones = await get_active_zones()
    return zones


@app.get("/zones/{zone_id}")
async def fetch_zone(zone_id: int) -> Dict:
    zone = await get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone


@app.post("/alerts/check")
async def trigger_alert_check(service: PriceAlertService = Depends(active_alerts)) -> JSONResponse:
    await service.manual_check()
    return JSONResponse({"status": "scheduled"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.fastapi_host, port=settings.fastapi_port, reload=settings.test_mode)
