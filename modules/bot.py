"""Telegram bot handlers for chart ingestion and user interaction."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (Application, ApplicationBuilder, CommandHandler,
                          ContextTypes, MessageHandler, filters)

from config import settings
from modules.agent import TradingAgent
from tools.alerts import PriceAlertService
from tools.logger import get_active_zones, log_zone
from tools.knowledge import chunk_text, read_document

LOGGER = logging.getLogger(__name__)


class TelegramBot:
    """Encapsulates telegram bot setup and handlers."""

    def __init__(self, agent: TradingAgent, uploads_dir: Path) -> None:
        self._agent = agent
        self._uploads_dir = uploads_dir
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._docs_dir = Path(settings.knowledge_path)
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir = Path(settings.audio_path)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._alert_service: Optional[PriceAlertService] = None
        self._application: Optional[Application] = None

    def build(self) -> Application:
        """Instantiate the telegram application with handlers."""
        if not settings.telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        self._application = ApplicationBuilder().token(settings.telegram_token).build()
        self._application.add_handler(CommandHandler("start", self._start))
        self._application.add_handler(CommandHandler("zones", self._zones))
        self._application.add_handler(CommandHandler("check", self._manual_check))
        self._application.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self._application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self._application.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        return self._application

    def attach_alert_service(self, service: PriceAlertService) -> None:
        """Attach the price alert service after bot construction."""
        self._alert_service = service

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Send an MT4/MT5 chart screenshot to receive Smart Money analysis."
        )

    async def _zones(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        zones = await get_active_zones()
        chat_id = update.effective_chat.id if update.effective_chat else None
        formatted = [
            f"#{zone['id']} {zone['pair']} {zone['bias']} entry {zone['entry']:.5f}"
            for zone in zones
            if not chat_id or zone.get("chat_id") == chat_id
        ]
        if not formatted:
            await update.message.reply_text("No active zones recorded.")
            return
        message = "\n".join(formatted)
        await update.message.reply_text(message)

    async def _manual_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._alert_service:
            await update.message.reply_text("Alert service not available.")
            return
        await self._alert_service.manual_check()
        await update.message.reply_text("Manual price check triggered.")

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        user = update.effective_user
        chat = update.effective_chat
        message = update.message.text
        metadata = {
            "user_id": user.id if user else None,
            "chat_id": chat.id if chat else None,
        }
        try:
            reply = await self._agent.converse(message, metadata)
            await update.message.reply_text(reply)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process text message", exc_info=exc)
            await update.message.reply_text("Unable to process request right now.")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return
        user = update.effective_user
        chat = update.effective_chat
        photo = update.message.photo[-1]
        file = await photo.get_file()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.file_id}.jpg"
        filepath = self._uploads_dir / filename
        await file.download_to_drive(custom_path=str(filepath))
        LOGGER.info("Image saved", extra={"path": str(filepath)})
        try:
            image_bytes = filepath.read_bytes()
            metadata = {
                "user_id": user.id if user else None,
                "chat_id": chat.id if chat else None,
                "caption": update.message.caption,
            }
            analysis = await self._agent.analyze_chart(image_bytes, update.message.caption, metadata)
            payload = analysis.to_dict()
            payload.update({
                "user_id": metadata["user_id"],
                "chat_id": metadata["chat_id"],
                "status": "active",
            })
            zone_id = await log_zone(payload, screenshot_path=str(filepath))
            memory_meta = {
                "user_id": metadata["user_id"],
                "chat_id": metadata["chat_id"],
                "screenshot_path": str(filepath),
                "caption": update.message.caption,
            }
            await self._agent.remember_zone(zone_id, analysis, memory_meta)
            response = self._format_response(zone_id, payload)
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process image", exc_info=exc)
            await update.message.reply_text("Unable to analyze chart at this time.")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document:
            return
        document = update.message.document
        user = update.effective_user
        filename = document.file_name or f"document_{document.file_unique_id}"
        suffix = (Path(filename).suffix or ".txt").lower()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest_path = self._docs_dir / f"{timestamp}_{document.file_unique_id}{suffix}"
        try:
            file = await document.get_file()
            await file.download_to_drive(custom_path=str(dest_path))
            if suffix == ".csv":
                result = await self._agent.analyze_csv(dest_path)
                await update.message.reply_text(result)
                return
            text = read_document(dest_path)
            chunks = chunk_text(text)
            if not chunks:
                await update.message.reply_text("Document contained no readable content.")
                return
            stored = await self._agent.ingest_document(
                title=filename,
                chunks=chunks,
                metadata={
                    "source_path": str(dest_path),
                    "uploaded_by": user.id if user else None,
                },
            )
            await update.message.reply_text(
                f"Indexed {stored} knowledge chunks from {filename}."
            )
        except ValueError as exc:
            await update.message.reply_text(str(exc))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to ingest document", exc_info=exc)
            await update.message.reply_text("Unable to process the document.")

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.voice:
            return
        voice = update.message.voice
        user = update.effective_user
        chat = update.effective_chat
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest_path = self._audio_dir / f"voice_{timestamp}_{voice.file_unique_id}.ogg"
        try:
            file = await voice.get_file()
            await file.download_to_drive(custom_path=str(dest_path))
            transcript = await self._agent.transcribe_voice(dest_path)
            if not transcript:
                await update.message.reply_text("Could not transcribe the voice message.")
                return
            metadata = {
                "user_id": user.id if user else None,
                "chat_id": chat.id if chat else None,
                "input_mode": "voice",
            }
            reply = await self._agent.converse(transcript, metadata)
            await update.message.reply_text(
                f"🗣️ {transcript}\n\n{reply}"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Voice handling failed", exc_info=exc)
            await update.message.reply_text("Unable to process the voice message right now.")

    @staticmethod
    def _format_response(zone_id: int, data: dict) -> str:
        return (
            f"**Zone #{zone_id}**\n"
            f"Pair: `{data['pair']}`\n"
            f"Bias: `{data['bias']}`\n"
            f"Entry: `{data['entry']:.5f}` | Stop: `{data['stop']:.5f}`\n"
            f"Target: `{data['target']:.5f}`\n"
            f"Type: `{data['zone_type']}` | EQ: `{data['eq_alignment']}`\n"
            f"Enhancer Score: `{data['enhancer_score']}`\n"
            f"Comment: {data['comment']}"
        )

    async def run_polling(self) -> None:
        if not self._application:
            self.build()
        assert self._application is not None
        await self._application.initialize()
        await self._application.start()
        LOGGER.info("Telegram bot polling started")
        try:
            if self._application.updater:
                await self._application.updater.start_polling()
        finally:
            await self._application.stop()
            await self._application.shutdown()

    def application(self) -> Application:
        if not self._application:
            self.build()
        assert self._application is not None
        return self._application
