import logging
from typing import Callable
from proxima_ops.control.permissions import Permissions, PermissionDenied

logger = logging.getLogger("proxima_ops.control.router")


class CommandRouter:
    def __init__(self, permissions: Permissions):
        self._commands: dict[str, Callable] = {}
        self._permissions = permissions

    def register(self, command: str, handler: Callable):
        self._commands[command.lower()] = handler
        logger.debug(f"Registered command: /{command}")

    def route(self, command: str, update) -> str:
        cmd = command.lower().split("@")[0].strip()
        handler = self._commands.get(cmd)
        if handler is None:
            return f"Unknown command: /{command}. Available: {', '.join(self._commands.keys())}"
        try:
            self._permissions.authorize(update)
        except PermissionDenied as e:
            return str(e)
        try:
            text = update.message.text if update.message else ""
            args = text.split()[1:] if text else []
            return handler(args, update)
        except Exception as e:
            logger.error(f"Command /{cmd} failed: {e}")
            return f"Error executing /{cmd}: {str(e)}"

    @property
    def available_commands(self) -> list[str]:
        return list(self._commands.keys())
