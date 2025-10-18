"""Fly.io entrypoint for the Telegram ↔ OpenAI assistant."""

import threading
from flask import Flask

from telegram_assistant_bridge import main as run_bridge


def _keepalive_server() -> None:
    """Start a tiny Flask server to satisfy Fly health checks."""
    app = Flask(__name__)

    @app.route("/")
    def ping() -> str:  # pragma: no cover
        return "ok"

    # Keeps Fly's TCP check happy without blocking the bot.
    app.run(host="0.0.0.0", port=8000)


def main() -> None:
    threading.Thread(target=_keepalive_server, daemon=True).start()
    run_bridge()


if __name__ == "__main__":
    main()
