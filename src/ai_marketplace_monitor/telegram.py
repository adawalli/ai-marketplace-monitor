import re
from dataclasses import dataclass
from logging import Logger
from typing import ClassVar, List

from .notification import PushNotificationConfig


@dataclass
class TelegramNotificationConfig(PushNotificationConfig):
    notify_method = "telegram"
    required_fields: ClassVar[List[str]] = ["telegram_bot_token", "telegram_chat_id"]

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    message_format: str | None = None

    def handle_telegram_bot_token(self: "TelegramNotificationConfig") -> None:
        if self.telegram_bot_token is None:
            return
        if not isinstance(self.telegram_bot_token, str) or not self.telegram_bot_token.strip():
            raise ValueError("telegram_bot_token must be a non-empty string")

        # Telegram bot tokens follow pattern: {bot_id}:{bot_token}
        # Example: 123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr
        stripped = self.telegram_bot_token.strip()
        if not re.match(r"^\d+:[A-Za-z0-9_-]+$", stripped):
            raise ValueError(
                "Invalid telegram bot token format. Expected format: 'bot_id:bot_token'"
            )

        self.telegram_bot_token = stripped

    def handle_telegram_chat_id(self: "TelegramNotificationConfig") -> None:
        if self.telegram_chat_id is None:
            return
        if not isinstance(self.telegram_chat_id, str) or not self.telegram_chat_id.strip():
            raise ValueError("telegram_chat_id must be a non-empty string")

        # Chat ID can be:
        # - Positive integer (user ID)
        # - Negative integer (group/channel ID)
        # - Username starting with @
        stripped = self.telegram_chat_id.strip()
        if stripped.startswith("@"):
            if len(stripped) < 2:
                raise ValueError("telegram_chat_id username must be at least 2 characters long")
            if not re.match(r"^@[a-zA-Z0-9_]+$", stripped):
                raise ValueError(
                    "telegram_chat_id username must contain only letters, numbers, and underscores"
                )
        else:
            # Should be a valid integer (positive or negative)
            try:
                int(stripped)
            except ValueError as e:
                raise ValueError(
                    "Invalid telegram chat ID format. Must be integer or username starting with @"
                ) from e

        self.telegram_chat_id = stripped

    def handle_message_format(self: "TelegramNotificationConfig") -> None:
        if self.message_format is None:
            self.message_format = "markdownv2"
        else:
            self.message_format = self.message_format.strip()

        valid_formats = ["plain_text", "markdown", "markdownv2", "html"]
        if self.message_format not in valid_formats:
            raise ValueError(f"Invalid message format. Must be one of {valid_formats}")

    def send_message(
        self: "TelegramNotificationConfig",
        title: str,
        message: str,
        logger: Logger | None = None,
    ) -> bool:
        """Send message via Telegram Bot API"""
        raise NotImplementedError("Telegram send_message implementation pending")
