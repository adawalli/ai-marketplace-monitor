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

    def _escape_markdownv2(self: "TelegramNotificationConfig", text: str) -> str:
        """Escape special characters for MarkdownV2 format while preserving formatting.

        MarkdownV2 requires escaping: _ * [ ] ( ) ~ ` > # + - = | { } . !
        But we need to preserve intentional formatting markup.
        """
        # Characters that need escaping in MarkdownV2
        special_chars = [
            "$",
            "!",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "+",
            "-",
            "=",
            "|",
            ".",
            "#",
            "~",
            ">",
            "<",
        ]

        escaped_text = text

        # Escape special characters but be careful about formatting
        for char in special_chars:
            escaped_text = escaped_text.replace(char, f"\\{char}")

        # Note: We intentionally don't escape _ * ` here as they are common formatting chars
        # A more sophisticated implementation would parse and preserve intentional formatting
        # while escaping standalone instances, but for now we let the user handle that

        return escaped_text

    def _escape_html(self: "TelegramNotificationConfig", text: str) -> str:
        """Escape special characters for HTML format while preserving valid tags.

        HTML requires escaping: < > &
        But we need to preserve valid HTML formatting tags.
        """
        import re

        # First, protect valid HTML tags that we want to preserve
        valid_tags = ["b", "i", "u", "s", "code", "pre", "a"]
        protected_tags = {}
        placeholder_counter = 0

        # Protect opening and closing valid tags
        for tag in valid_tags:
            # Protect opening tags (including those with attributes like <a href="...">)
            pattern = f"<{tag}(?:\\s[^>]*)?>|</{tag}>"
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                placeholder = f"__PROTECTED_TAG_{placeholder_counter}__"
                protected_tags[placeholder] = match
                text = text.replace(match, placeholder, 1)
                placeholder_counter += 1

        # Now escape all remaining HTML characters
        escaped_text = text
        escaped_text = escaped_text.replace("&", "&amp;")  # Must be first
        escaped_text = escaped_text.replace("<", "&lt;")
        escaped_text = escaped_text.replace(">", "&gt;")

        # Restore protected tags
        for placeholder, original_tag in protected_tags.items():
            escaped_text = escaped_text.replace(placeholder, original_tag)

        return escaped_text

    def _prepare_plain_text(
        self: "TelegramNotificationConfig", title: str, message: str
    ) -> tuple[str, None]:
        """Prepare message for plain text format (no escaping needed)."""
        combined_message = f"{title}\n\n{message}"
        return combined_message, None

    def _prepare_markdown(
        self: "TelegramNotificationConfig", title: str, message: str
    ) -> tuple[str, str]:
        """Prepare message for Markdown format (legacy format)."""
        # For legacy Markdown, we don't escape - let Telegram handle it
        combined_message = f"{title}\n\n{message}"
        return combined_message, "Markdown"

    def _prepare_markdownv2(
        self: "TelegramNotificationConfig", title: str, message: str
    ) -> tuple[str, str]:
        """Prepare message for MarkdownV2 format with proper escaping."""
        # Apply escaping to both title and message
        escaped_title = self._escape_markdownv2(title)
        escaped_message = self._escape_markdownv2(message)
        combined_message = f"{escaped_title}\n\n{escaped_message}"
        return combined_message, "MarkdownV2"

    def _prepare_html(
        self: "TelegramNotificationConfig", title: str, message: str
    ) -> tuple[str, str]:
        """Prepare message for HTML format with proper escaping."""
        # Apply HTML escaping to both title and message
        escaped_title = self._escape_html(title)
        escaped_message = self._escape_html(message)
        combined_message = f"{escaped_title}\n\n{escaped_message}"
        return combined_message, "HTML"

    def _prepare_message_for_format(
        self: "TelegramNotificationConfig", title: str, message: str
    ) -> tuple[str, str | None]:
        """Prepare message content based on the selected format."""
        if self.message_format == "plain_text":
            return self._prepare_plain_text(title, message)
        elif self.message_format == "markdown":
            return self._prepare_markdown(title, message)
        elif self.message_format == "markdownv2":
            return self._prepare_markdownv2(title, message)
        elif self.message_format == "html":
            return self._prepare_html(title, message)
        else:
            # Fallback to plain text for unknown formats
            return self._prepare_plain_text(title, message)

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

        # Define fallback sequence
        fallback_formats = []
        if self.message_format == "markdownv2":
            fallback_formats = ["html", "plain_text"]
        elif self.message_format == "html":
            fallback_formats = ["plain_text"]
        elif self.message_format == "markdown":
            fallback_formats = ["html", "plain_text"]
        # plain_text has no fallbacks

        # Define async function to send telegram message
        async def _send_telegram_message():
            bot = telegram.Bot(token=self.telegram_bot_token)
            await bot.send_message(
                chat_id=self.telegram_chat_id,
                text=current_msg_text,
                parse_mode=current_msg_parse_mode,
            )
            return True
            # Use asyncio.run to maintain synchronous interface

        # Try primary format first
        formats_to_try = [self.message_format, *fallback_formats]

        for attempt_format in formats_to_try:
            try:
                # Prepare message for this format
                if attempt_format == "plain_text":
                    current_msg_text, current_msg_parse_mode = self._prepare_plain_text(
                        title, message
                    )
                elif attempt_format == "markdown":
                    current_msg_text, current_msg_parse_mode = self._prepare_markdown(
                        title, message
                    )
                elif attempt_format == "markdownv2":
                    current_msg_text, current_msg_parse_mode = self._prepare_markdownv2(
                        title, message
                    )
                elif attempt_format == "html":
                    current_msg_text, current_msg_parse_mode = self._prepare_html(title, message)
                else:
                    # Unknown format, fallback to plain text
                    current_msg_text, current_msg_parse_mode = self._prepare_plain_text(
                        title, message
                    )

                # Execute async function synchronously
                result = asyncio.run(_send_telegram_message())

                # If we get here, the message was sent successfully
                if logger and attempt_format != self.message_format:
                    logger.warning(
                        f"Telegram message sent using fallback format '{attempt_format}' instead of '{self.message_format}'"
                    )

                return result

            except Exception as e:
                # Check if this is a formatting-related error that should trigger fallback
                error_message = str(e).lower()
                is_format_error = any(
                    keyword in error_message
                    for keyword in [
                        "parse",
                        "parsing",
                        "markdown",
                        "html",
                        "format",
                        "entity",
                        "tag",
                    ]
                )

                # If this is not a format error, or we're on the last format, don't continue
                if not is_format_error or attempt_format == formats_to_try[-1]:
                    if logger:
                        logger.error(f"Failed to send Telegram message: {e!s}")
                    return False

                # Continue to next format for format errors
                if logger:
                    logger.warning(
                        f"Telegram format '{attempt_format}' failed, trying fallback: {e!s}"
                    )
                continue

        # Should never reach here due to logic above, but safety fallback
        return False
