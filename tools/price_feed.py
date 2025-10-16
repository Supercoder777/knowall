"""External price feed integrations."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

import aiohttp
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings

LOGGER = logging.getLogger(__name__)


class AbstractPriceFeed(ABC):
    """Interface for async price feed implementations."""

    def __init__(self) -> None:
        self._cache: Dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def get_price(self, pair: str) -> Optional[float]:
        """Return latest price for the given pair with caching."""
        key = pair.upper()
        async with self._lock:
            cached = self._cache.get(key)
            if cached and (time.time() - cached[1]) < 30:
                return cached[0]
            price = await self._retrieve_price(key)
            if price is not None:
                self._cache[key] = (price, time.time())
            return price

    @abstractmethod
    async def _retrieve_price(self, pair: str) -> Optional[float]:
        """Fetch price from remote source."""


class BinancePriceFeed(AbstractPriceFeed):
    """Binance price feed leveraging public REST endpoints."""

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self) -> None:
        super().__init__()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        self._session = aiohttp.ClientSession()
        return self._session

    async def _retrieve_price(self, pair: str) -> Optional[float]:
        symbol = self._to_symbol(pair)
        LOGGER.debug("Fetching Binance price", extra={"pair": pair, "symbol": symbol})
        session = await self._ensure_session()

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        ):
            with attempt:
                async with session.get(f"{self.BASE_URL}/ticker/price", params={"symbol": symbol}, timeout=10) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    data = await resp.json()
                    price = float(data["price"])
                    return price
        return None

    @staticmethod
    def _to_symbol(pair: str) -> str:
        base = pair[:3]
        quote = pair[3:]
        if quote == "USD":
            quote = "USDT"
        return f"{base}{quote}"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class OandaPriceFeed(AbstractPriceFeed):
    """OANDA REST API price feed."""

    STREAM_URL = "https://api-fxtrade.oanda.com/v3/instruments/{instrument}/price"

    def __init__(self, api_key: Optional[str], account_id: Optional[str]) -> None:
        super().__init__()
        self._api_key = api_key
        self._account_id = account_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _retrieve_price(self, pair: str) -> Optional[float]:
        if not self._api_key or not self._account_id:
            LOGGER.warning("OANDA credentials missing; returning None")
            return None
        session = await self._ensure_session()
        url = self.STREAM_URL.format(instrument=self._format_instrument(pair))
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        ):
            with attempt:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    data = await resp.json()
                    bids = data.get("bids")
                    asks = data.get("asks")
                    if not bids or not asks:
                        return None
                    price = (
                        float(bids[0].get("price", 0)) + float(asks[0].get("price", 0))
                    ) / 2
                    return price
        return None

    @staticmethod
    def _format_instrument(pair: str) -> str:
        base = pair[:3]
        quote = pair[3:]
        return f"{base}_{quote}"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class TestPriceFeed(AbstractPriceFeed):
    """Deterministic price feed backed by in-memory map for tests."""

    def __init__(self, prices: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self._prices = prices or {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2725,
            "XAUUSD": 2350.25,
        }

    async def _retrieve_price(self, pair: str) -> Optional[float]:
        return self._prices.get(pair.upper())


async def get_price_feed() -> AbstractPriceFeed:
    """Factory returning configured price feed instance."""
    if settings.test_mode:
        return TestPriceFeed()
    if settings.price_provider == "oanda":
        return OandaPriceFeed(settings.oanda_api_key, settings.oanda_account_id)
    return BinancePriceFeed()


async def close_price_feed(feed: AbstractPriceFeed) -> None:
    """Close price feed resources if supported."""
    close_fn = getattr(feed, "close", None)
    if close_fn:
        await close_fn()
