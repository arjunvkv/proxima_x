"""
Telegram Bot Control Layer for Proxima Ops.

python-telegram-bot v22+ required: pip install python-telegram-bot
"""
from __future__ import annotations
import logging
import asyncio
from typing import Optional, TYPE_CHECKING
from proxima_ops.config.settings import SETTINGS
from proxima_ops.control.permissions import Permissions
from proxima_ops.control.command_router import CommandRouter

logger = logging.getLogger("proxima_ops.telegram")

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Install with: pip install python-telegram-bot")


class TelegramBot:
    def __init__(self, router: CommandRouter):
        self._router = router
        self._app: Optional[Application] = None
        self._running = False

    async def _handle_command(self, update: "Update", context: object):
        if update.message is None:
            return
        cmd = update.message.text.split()[0][1:] if update.message.text else ""
        response = self._router.route(cmd, update)
        await update.message.reply_text(response)

    def _build_app(self) -> Optional[Application]:
        if not TELEGRAM_AVAILABLE:
            logger.error("python-telegram-bot not available")
            return None
        if not SETTINGS.telegram_token:
            logger.error("TELEGRAM_TOKEN not configured")
            return None
        app = Application.builder().token(SETTINGS.telegram_token).build()
        for cmd in self._router.available_commands:
            app.add_handler(CommandHandler(cmd, self._handle_command))
        return app

    def start(self):
        self._app = self._build_app()
        if self._app is None:
            logger.error("Cannot start Telegram bot — configuration missing")
            return
        logger.info("Starting Telegram bot polling...")
        self._app.run_polling(allowed_updates=["message"])

    def stop(self):
        if self._app:
            self._app.stop()

    async def send_message(self, text: str):
        if not TELEGRAM_AVAILABLE or not self._app:
            return
        chat_id = SETTINGS.telegram_chat_id
        if not chat_id:
            return
        try:
            await self._app.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def send_sync(self, text: str):
        try:
            asyncio.run(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_message(text))
            loop.close()
