"""Telegram ↔ OpenAI Assistant bridge with function calling and stateful features."""

# Fly.io secrets setup commands:
# flyctl secrets set TELEGRAM_BOT_TOKEN=your_token
# flyctl secrets set OPENAI_API_KEY=your_key
# flyctl secrets set ASSISTANT_ID=asst_xxxx

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiofiles
import aiosqlite
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
)
from openai.types import FileObject
from openai.types.beta.threads import Message
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# -------------------------------------------------------
# Environment setup
# -------------------------------------------------------
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Telegram Assistant")
ASSISTANT_INSTRUCTIONS = os.getenv(
    "ASSISTANT_INSTRUCTIONS",
    "You are a helpful trading assistant. Keep replies concise and actionable.",
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

if not BOT_TOKEN:
    raise RuntimeError(
        "Missing TELEGRAM_BOT_TOKEN. Set it via Fly secrets: flyctl secrets set TELEGRAM_BOT_TOKEN=your_token"
    )
if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it via Fly secrets: flyctl secrets set OPENAI_API_KEY=your_key"
    )
if not ASSISTANT_ID:
    raise RuntimeError(
        "Missing ASSISTANT_ID. Set it via Fly secrets: flyctl secrets set ASSISTANT_ID=asst_xxxx"
    )

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
LOGGER = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DB_PATH = Path("data") / "assistant_state.db"
DB_LOCK = asyncio.Lock()
DB_INITIALISED = False

DEFAULT_MODE = "trading"
MODE_PRESETS: Dict[str, Dict[str, str]] = {
    "trading": {
        "instructions": "You are a trading assistant. Share concise, risk-aware insights.",
        "model": ASSISTANT_MODEL,
    },
    "engineering": {
        "instructions": "You are a senior software engineer. Provide detailed, pragmatic guidance.",
        "model": ASSISTANT_MODEL,
    },
    "analysis": {
        "instructions": "You are an analytical assistant. Offer structured, data-driven summaries.",
        "model": ASSISTANT_MODEL,
    },
}

USER_LOCKS: Dict[str, asyncio.Lock] = {}
USER_LOCKS_LOCK = asyncio.Lock()

SENSITIVE_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9]{10,}"), "[redacted-key]"),
    (re.compile(r"\b\d{9,}:[A-Za-z0-9_-]{20,}\b"), "[redacted-bot-token]"),
]

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@dataclass
class AssistantReply:
    """Container for assistant responses."""

    texts: List[str] = field(default_factory=list)
    file_ids: List[str] = field(default_factory=list)


async def init_db() -> None:
    """Initialise the SQLite database if required."""
    global DB_INITIALISED
    if DB_INITIALISED:
        return
    async with DB_LOCK:
        if DB_INITIALISED:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            mode_default = DEFAULT_MODE.replace("'", "''")
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS user_threads (
                    user_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT '{mode_default}',
                    model TEXT,
                    instructions TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.commit()
        DB_INITIALISED = True


async def get_thread(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch stored thread and preferences for a user."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_id, thread_id, mode, model, instructions FROM user_threads WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def save_thread(
    user_id: str,
    thread_id: str,
    *,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
) -> None:
    """Persist or update a user's thread state."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_threads (user_id, thread_id, mode, model, instructions)
            VALUES (?, ?, COALESCE(?, ?), ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                thread_id = excluded.thread_id,
                mode = COALESCE(excluded.mode, user_threads.mode),
                model = COALESCE(excluded.model, user_threads.model),
                instructions = COALESCE(excluded.instructions, user_threads.instructions),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                user_id,
                thread_id,
                mode,
                DEFAULT_MODE,
                model,
                instructions,
            ),
        )
        await db.commit()


async def delete_thread(user_id: str) -> None:
    """Remove stored state for a user."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM user_threads WHERE user_id = ?", (user_id,))
        await db.commit()


async def update_user_preferences(
    user_id: str,
    *,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
) -> None:
    """Update stored preferences while keeping the existing thread."""
    state = await get_thread(user_id)
    if not state:
        return
    await save_thread(
        user_id,
        state["thread_id"],
        mode=mode or state.get("mode"),
        model=model if model is not None else state.get("model"),
        instructions=instructions if instructions is not None else state.get("instructions"),
    )


def _redact(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    redacted = value
    for pattern, replacement in SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _log(level: int, message: str, **context: Any) -> None:
    context = {key: _redact(str(val)) for key, val in context.items()}
    LOGGER.log(level, message, extra=context)


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    async with USER_LOCKS_LOCK:
        if user_id not in USER_LOCKS:
            USER_LOCKS[user_id] = asyncio.Lock()
        return USER_LOCKS[user_id]


def _resolve_user_mode(state: Dict[str, Any]) -> Dict[str, str]:
    mode = (state.get("mode") or DEFAULT_MODE).lower()
    preset = MODE_PRESETS.get(mode, MODE_PRESETS[DEFAULT_MODE])
    resolved = {
        "mode": mode,
        "model": state.get("model") or preset["model"],
        "instructions": state.get("instructions") or preset["instructions"],
    }
    return resolved


async def _get_or_create_thread(user_id: str) -> Dict[str, Any]:
    state = await get_thread(user_id)
    if state:
        return state

    await _ensure_assistant_exists()
    thread = await client.beta.threads.create()
    new_state = {
        "user_id": user_id,
        "thread_id": thread.id,
        "mode": DEFAULT_MODE,
        "model": MODE_PRESETS[DEFAULT_MODE]["model"],
        "instructions": MODE_PRESETS[DEFAULT_MODE]["instructions"],
    }
    await save_thread(
        user_id,
        new_state["thread_id"],
        mode=new_state["mode"],
        model=new_state["model"],
        instructions=new_state["instructions"],
    )
    _log(
        logging.INFO,
        "Created new thread",
        user_id=user_id,
        thread_id=new_state["thread_id"],
    )
    return new_state


async def _ensure_assistant_exists() -> None:
    """Validate that the configured assistant exists and supports only allowed tools."""
    try:
        assistant = await client.beta.assistants.retrieve(assistant_id=ASSISTANT_ID)
    except NotFoundError as exc:  # noqa: BLE001
        _log(
            logging.ERROR,
            "Configured assistant not found; update ASSISTANT_ID secret",
            assistant_id=ASSISTANT_ID,
        )
        raise RuntimeError(
            "ASSISTANT_ID does not match an existing assistant. Update the Fly secret with a valid ID."
        ) from exc

    tools = getattr(assistant, "tools", None) or []
    if tools:
        allowed = {"code_interpreter", "file_search", "retrieval"}
        unsupported = [t for t in tools if getattr(t, "type", None) not in allowed]

        if unsupported:
            _log(
                logging.ERROR,
                "Assistant has unsupported tools configured",
                assistant_id=ASSISTANT_ID,
                tools=[getattr(t, "type", str(t)) for t in unsupported],
            )
            raise RuntimeError(
                f"Unsupported tools detected: {[getattr(t, 'type', str(t)) for t in unsupported]}"
            )
        _log(
            logging.INFO,
            "Assistant tools accepted",
            assistant_id=ASSISTANT_ID,
            tools=[getattr(t, "type", str(t)) for t in tools],
        )
    else:
        _log(logging.INFO, "Assistant has no tools configured", assistant_id=ASSISTANT_ID)


async def _handle_tool_calls(
    user_id: str,
    thread_id: str,
    run_id: str,
    status: Any,
) -> None:
    required = getattr(status, "required_action", None)
    if not required or not getattr(required, "submit_tool_outputs", None):
        return
    tool_calls = required.submit_tool_outputs.tool_calls
    outputs = []
    for call in tool_calls:
        function = getattr(call, "function", None)
        if not function:
            continue
        try:
            result = await run_local_function(function.name, function.arguments)
            outputs.append({"tool_call_id": call.id, "output": result})
            _log(
                logging.INFO,
                "Executed tool call",
                user_id=user_id,
                thread_id=thread_id,
                run_id=run_id,
                tool=function.name,
            )
        except Exception as exc:  # noqa: BLE001
            _log(
                logging.ERROR,
                "Tool execution failed",
                user_id=user_id,
                thread_id=thread_id,
                run_id=run_id,
                tool=function.name,
                error=str(exc),
            )
            outputs.append({"tool_call_id": call.id, "output": f"Error: {exc}"})
    if outputs:
        await client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=outputs,
        )


async def run_local_function(name: str, arguments: str) -> str:
    """Dispatch a local tool by name."""
    try:
        parsed_args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        parsed_args = {}
    func = LOCAL_TOOL_REGISTRY.get(name)
    if not func:
        raise ValueError(f"Unknown tool {name}")
    if asyncio.iscoroutinefunction(func):
        result = await func(**parsed_args)
    else:
        result = func(**parsed_args)
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


async def tool_get_market_data(symbol: str = "BTCUSDT") -> str:
    """Mock market data lookup."""
    return f"[mock] Latest price for {symbol}: 123.45"


async def tool_summarize_file(file_id: str) -> str:
    """Mock file summary."""
    return f"[mock] Summary for file {file_id}: Unable to read in mock environment."


async def tool_plot_csv(file_id: str, x: str = "time", y: str = "value") -> str:
    """Mock plot generation."""
    return f"[mock] Generated plot for {file_id} with {x} vs {y}."


LOCAL_TOOL_REGISTRY = {
    "get_market_data": tool_get_market_data,
    "summarize_file": tool_summarize_file,
    "plot_csv": tool_plot_csv,
}


async def _run_assistant_and_fetch_reply(
    user_id: str,
    state: Dict[str, Any],
    *,
    poll_interval: float = 1.0,
    timeout: float = 180.0,
) -> tuple[AssistantReply, Any]:
    thread_id = state["thread_id"]
    params = _resolve_user_mode(state)
    await _ensure_assistant_exists()
    try:
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            model=params["model"],
            instructions=params["instructions"],
        )
    except APIError as exc:
        _log(
            logging.ERROR,
            "Failed to start run",
            user_id=user_id,
            thread_id=thread_id,
            error=str(exc),
        )
        raise

    _log(
        logging.INFO,
        "Assistant run started",
        user_id=user_id,
        thread_id=thread_id,
        run_id=run.id,
        mode=params["mode"],
        model=params["model"],
    )

    loop = asyncio.get_running_loop()
    start = loop.time()
    current_status = None

    while True:
        if loop.time() - start > timeout:
            raise TimeoutError("Timed out waiting for assistant reply")
        try:
            current_status = await client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )
        except (APIConnectionError, APITimeoutError) as exc:
            _log(
                logging.WARNING,
                "Transient error while polling run",
                user_id=user_id,
                thread_id=thread_id,
                run_id=run.id,
                error=str(exc),
            )
            await asyncio.sleep(poll_interval)
            continue

        _log(
            logging.DEBUG,
            "Assistant run status",
            user_id=user_id,
            thread_id=thread_id,
            run_id=run.id,
            status=current_status.status,
        )

        if current_status.status == "completed":
            break
        if current_status.status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"Assistant run ended with status {current_status.status}")
        if current_status.status == "requires_action":
            await _handle_tool_calls(user_id, thread_id, run.id, current_status)
        await asyncio.sleep(poll_interval)

    messages = await client.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc",
        limit=20,
    )

    reply = _extract_assistant_reply(messages.data, current_status)
    if reply.texts or reply.file_ids:
        return reply, run
    raise RuntimeError("Assistant returned no reply")


def _extract_assistant_reply(messages: List[Message], status: Any) -> AssistantReply:
    reply = AssistantReply()
    for message in messages:
        if message.role != "assistant":
            continue
        for item in message.content:
            item_type = getattr(item, "type", None)
            if item_type == "text" and hasattr(item, "text"):
                reply.texts.append(item.text.value)
            elif item_type == "image_file" and hasattr(item, "image_file"):
                reply.file_ids.append(item.image_file.file_id)
            elif item_type == "file" and hasattr(item, "file"):
                reply.file_ids.append(item.file.file_id)
        if reply.texts or reply.file_ids:
            break
    for file_ref in getattr(status, "output_files", []) or []:
        if file_ref.id not in reply.file_ids:
            reply.file_ids.append(file_ref.id)
    return reply


async def _download_openai_file(file_id: str) -> tuple[bytes, str, str]:
    metadata: FileObject = await client.files.retrieve(file_id)
    stream = await client.files.content(file_id)
    data = await stream.aread()
    filename = metadata.filename or f"{file_id}.bin"
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return data, filename, mime_type


def _build_file_path(user_id: str, filename: str) -> Path:
    safe_name = filename or "attachment"
    unique = uuid4().hex
    return UPLOAD_DIR / f"{user_id}_{unique}_{safe_name}"


async def _prepare_openai_upload(user_id: str, telegram_file, filename: str) -> BytesIO:
    path = _build_file_path(user_id, filename)
    await telegram_file.download_to_drive(str(path))
    async with aiofiles.open(path, "rb") as handle:
        data = await handle.read()
    path.unlink(missing_ok=True)
    buffer = BytesIO(data)
    buffer.name = filename
    return buffer


async def _forward_contents_to_assistant(
    update: Update,
    user_id: str,
    contents: List[Dict[str, Any]],
) -> None:
    state = await _get_or_create_thread(user_id)
    thread_id = state["thread_id"]
    log_context = {"user_id": user_id, "thread_id": thread_id}
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=contents,
    )
    try:
        reply, run = await _run_assistant_and_fetch_reply(user_id, state)
    except BadRequestError as exc:
        message = str(exc)
        _log(logging.ERROR, "OpenAI rejected message", **log_context, error=message)
        await update.message.reply_text(
            "Assistant could not process the request right now. Please try again shortly."
        )
        return
    except (APIConnectionError, APITimeoutError) as exc:
        _log(logging.ERROR, "Network error while fetching reply", **log_context, error=str(exc))
        await update.message.reply_text("Network error communicating with assistant. Please retry.")
        return
    except Exception as exc:  # noqa: BLE001
        _log(logging.ERROR, "Failed to fetch assistant reply", **log_context, error=str(exc))
        await update.message.reply_text("Unable to process your request right now.")
        return

    run_id = getattr(run, "id", "")
    _log(logging.INFO, "Sending assistant reply", **log_context, run_id=run_id)

    if reply.texts:
        text_reply = "\n\n".join(reply.texts).strip()
        if text_reply:
            await update.message.reply_text(text_reply)

    for file_id in reply.file_ids:
        try:
            data, filename, mime_type = await _download_openai_file(file_id)
            buffer = BytesIO(data)
            buffer.name = filename
            if mime_type.startswith("image/"):
                await update.message.reply_photo(photo=buffer)
            else:
                await update.message.reply_document(document=buffer, filename=filename)
            _log(logging.INFO, "Sent assistant file", **log_context, run_id=run_id, file_id=file_id)
        except Exception as exc:  # noqa: BLE001
            _log(logging.ERROR, "Failed to send assistant file", **log_context, file_id=file_id, error=str(exc))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "🤖 Hey! Send me text, images, or documents and I'll forward them to your OpenAI assistant.\n"
        "Use /mode <trading|engineering|analysis> to switch modes, and /status to inspect your session."
    )


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Switch assistant mode for the user."""
    if not update.message:
        return
    user_id = str(update.effective_user.id)
    args = context.args if context.args else []
    if not args:
        await update.message.reply_text(
            "Please provide a mode. Available: " + ", ".join(sorted(MODE_PRESETS.keys()))
        )
        return
    mode = args[0].lower()
    if mode not in MODE_PRESETS:
        await update.message.reply_text(
            f"Unknown mode '{mode}'. Available: " + ", ".join(sorted(MODE_PRESETS.keys()))
        )
        return
    async with (await _get_user_lock(user_id)):
        state = await _get_or_create_thread(user_id)
        preset = MODE_PRESETS[mode]
        await save_thread(
            user_id,
            state["thread_id"],
            mode=mode,
            model=preset["model"],
            instructions=preset["instructions"],
        )
    await update.message.reply_text(f"Mode updated to {mode}.")


async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report current state for the user."""
    if not update.message:
        return
    user_id = str(update.effective_user.id)
    lock = await _get_user_lock(user_id)
    async with lock:
        state = await get_thread(user_id)
    if not state:
        await update.message.reply_text("No active session yet. Send a message to get started.")
        return
    params = _resolve_user_mode(state)
    status_lines = [
        f"Thread ID: {state['thread_id']}",
        f"Mode: {params['mode']}",
        f"Model: {params['model']}",
    ]
    instructions_preview = params["instructions"][:200].strip()
    if instructions_preview:
        status_lines.append(f"Instructions: {instructions_preview}...")
    status_lines.append(f"Run in progress: {'yes' if lock.locked() else 'no'}")
    await update.message.reply_text("\n".join(status_lines))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    user_id = str(update.effective_user.id)
    text = update.message.text
    async with (await _get_user_lock(user_id)):
        _log(logging.INFO, "Received text", user_id=user_id)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        contents = [{"type": "text", "text": text}]
        await _forward_contents_to_assistant(update, user_id, contents)


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    user_id = str(update.effective_user.id)
    async with (await _get_user_lock(user_id)):
        file_entry = None
        caption = message.caption or "Please analyze this file."
        if message.photo:
            file_entry = message.photo[-1]
            filename = f"photo_{file_entry.file_unique_id}.jpg"
            mime_type = "image/jpeg"
        elif message.document:
            file_entry = message.document
            filename = file_entry.file_name or f"document_{file_entry.file_unique_id}"
            mime_type = file_entry.mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        else:
            return
        telegram_file = await file_entry.get_file()
        buffer = await _prepare_openai_upload(user_id, telegram_file, filename)
        upload = await client.files.create(
            file=buffer,
            purpose="assistants",
            filename=filename,
        )
        content_block: Dict[str, Any]
        if mime_type.startswith("image/"):
            content_block = {"type": "image_file", "image_file": {"file_id": upload.id}}
        else:
            content_block = {"type": "file", "file": {"file_id": upload.id}}
        contents = [{"type": "text", "text": caption}, content_block]
        _log(logging.INFO, "Received file", user_id=user_id, file_id=upload.id, mime=mime_type)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
        await _forward_contents_to_assistant(update, user_id, contents)


async def _post_init(application) -> None:
    await init_db()
    await _ensure_assistant_exists()


def main() -> None:
    print("Starting Telegram ↔ Assistant bridge...")
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(_post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("mode", set_mode))
    application.add_handler(CommandHandler("status", show_status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(
        MessageHandler((filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_file)
    )

    application.run_polling()


# Fly.io deployment checklist:
# 1. Set secrets: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, ASSISTANT_ID
# 2. Redeploy: flyctl deploy --remote-only
# 3. Monitor logs: flyctl logs -a telegram-assistant
# 4. Check status: flyctl status

if __name__ == "__main__":
    main()
