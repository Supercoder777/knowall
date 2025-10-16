"""Async SQLite logging utilities for zone tracking."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from pathlib import Path

from sqlalchemy import (DateTime, Float, Integer, MetaData, String, Text,
                       UniqueConstraint, select, update)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import (AsyncAttrs, AsyncEngine, AsyncSession,
                                   async_sessionmaker, create_async_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import settings


def _build_engine() -> AsyncEngine:
    """Create async SQLAlchemy engine using configured database URL."""
    prefix = "sqlite+aiosqlite:///"
    if settings.database_url.startswith(prefix):
        db_path = settings.database_url[len(prefix):]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(settings.database_url, echo=False, future=True)


metadata = MetaData()


class Base(AsyncAttrs, DeclarativeBase):
    """Declarative base class bound to shared metadata."""

    metadata = metadata


class ZoneTracker(Base):
    """ORM model for tracked trading zones."""

    __tablename__ = "zone_tracker"
    __table_args__ = (
        UniqueConstraint("chat_id", "pair", "entry", name="uq_zone_chat_pair_entry"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chat_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pair: Mapped[str] = mapped_column(String(16), nullable=False)
    bias: Mapped[str] = mapped_column(String(8), nullable=False)
    entry: Mapped[float] = mapped_column(Float, nullable=False)
    stop: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    zone_type: Mapped[str] = mapped_column(String(16), nullable=False)
    eq_alignment: Mapped[str] = mapped_column(String(8), nullable=False)
    enhancer_score: Mapped[float] = mapped_column(Float, nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    screenshot_path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


_engine: AsyncEngine = _build_engine()
_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine, expire_on_commit=False
)


async def init_db() -> None:
    """Create database tables if they do not yet exist."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Provide an async transactional scope."""
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def log_zone(entry_data: Dict[str, Any], screenshot_path: Optional[str] = None) -> int:
    """Insert or update a trading zone entry.

    Returns the primary key of the stored record.
    """
    payload = {
        **entry_data,
        "screenshot_path": screenshot_path,
        "updated_at": datetime.now(timezone.utc),
    }
    payload.setdefault("status", "active")
    stmt = sqlite_insert(ZoneTracker).values(payload)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[ZoneTracker.chat_id, ZoneTracker.pair, ZoneTracker.entry],
        set_=payload,
    )

    async with session_scope() as session:
        result = await session.execute(upsert_stmt.returning(ZoneTracker.id))
        zone_id = result.scalar_one()
    return zone_id


async def update_zone_status(zone_id: int, status: str) -> None:
    """Update the status field for a zone."""
    stmt = (
        update(ZoneTracker)
        .where(ZoneTracker.id == zone_id)
        .values(status=status, updated_at=datetime.now(timezone.utc))
    )
    async with session_scope() as session:
        await session.execute(stmt)


async def get_active_zones() -> List[Dict[str, Any]]:
    """Fetch all active zones as dictionaries."""
    stmt = select(ZoneTracker).where(ZoneTracker.status == "active")
    async with session_scope() as session:
        results = await session.execute(stmt)
        zones = [row.ZoneTracker for row in results.fetchall()]
    return [
        {
            "id": zone.id,
            "user_id": zone.user_id,
            "chat_id": zone.chat_id,
            "pair": zone.pair,
            "bias": zone.bias,
            "entry": zone.entry,
            "stop": zone.stop,
            "target": zone.target,
            "zone_type": zone.zone_type,
            "eq_alignment": zone.eq_alignment,
            "enhancer_score": zone.enhancer_score,
            "comment": zone.comment,
            "status": zone.status,
            "screenshot_path": zone.screenshot_path,
            "created_at": zone.created_at,
            "updated_at": zone.updated_at,
        }
        for zone in zones
    ]


async def get_zone(zone_id: int) -> Optional[Dict[str, Any]]:
    """Return a single zone by identifier."""
    stmt = select(ZoneTracker).where(ZoneTracker.id == zone_id)
    async with session_scope() as session:
        result = await session.execute(stmt)
        zone = result.scalar_one_or_none()
    if zone is None:
        return None
    return {
        "id": zone.id,
        "user_id": zone.user_id,
        "chat_id": zone.chat_id,
        "pair": zone.pair,
        "bias": zone.bias,
        "entry": zone.entry,
        "stop": zone.stop,
        "target": zone.target,
        "zone_type": zone.zone_type,
        "eq_alignment": zone.eq_alignment,
        "enhancer_score": zone.enhancer_score,
        "comment": zone.comment,
        "status": zone.status,
        "screenshot_path": zone.screenshot_path,
        "created_at": zone.created_at,
        "updated_at": zone.updated_at,
    }


async def close_engine() -> None:
    """Dispose engine connections for shutdown."""
    await _engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
