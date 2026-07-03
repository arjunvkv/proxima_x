import logging
from proxima_ops.config.settings import SETTINGS

logger = logging.getLogger("proxima_ops.control.auth")


class PermissionDenied(Exception):
    pass


class Permissions:
    def __init__(self):
        self._authorized_users = SETTINGS.telegram_auth_users

    def check_user(self, user_id: int) -> bool:
        if not self._authorized_users:
            logger.warning("No authorized users configured — allowing all")
            return True
        return user_id in self._authorized_users

    def authorize(self, update):
        user = update.effective_user
        if user is None:
            raise PermissionDenied("No user info available")
        if not self.check_user(user.id):
            logger.warning(f"Unauthorized access attempt by user {user.id} ({user.username})")
            raise PermissionDenied(f"User {user.id} not authorized")
        return user
