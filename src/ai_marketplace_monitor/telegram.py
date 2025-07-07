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
        import asyncio

        import telegram

        # Parameter validation - check title first
        if not isinstance(title, str) or not title or not title.strip():
            raise ValueError("title must be a non-empty string")

        # Parameter validation - check message second
        if not isinstance(message, str) or not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        # Trim whitespace from parameters
        title = title.strip()
        message = message.strip()

        # Configuration validation
        if not self.telegram_bot_token or not self.telegram_bot_token.strip():
            raise ValueError("telegram_bot_token is required")

        if not self.telegram_chat_id or not self.telegram_chat_id.strip():
            raise ValueError("telegram_chat_id is required")

        # Validate message format
        valid_formats = ["plain_text", "markdown", "markdownv2", "html"]
        if self.message_format not in valid_formats:
            raise ValueError(f"Invalid message format. Must be one of {valid_formats}")

        # Combine title and message
        combined_message = f"{title}\n\n{message}"

        # Convert message format for telegram API
        parse_mode = None
        if self.message_format == "markdown":
            parse_mode = "Markdown"
        elif self.message_format == "markdownv2":
            parse_mode = "MarkdownV2"
        elif self.message_format == "html":
            parse_mode = "HTML"
        # plain_text uses None (no parse_mode)

        try:
            # Create async function to send message
            async def _send_telegram_message():
                bot = telegram.Bot(token=self.telegram_bot_token)
                await bot.send_message(
                    chat_id=self.telegram_chat_id, text=combined_message, parse_mode=parse_mode
                )
                return True

            # Use asyncio.run to execute async function synchronously
            result = asyncio.run(_send_telegram_message())
            return result

        except Exception as e:
            # Log error if logger is provided
            if logger:
                logger.error(f"Failed to send Telegram message: {e!s}")
            return False
