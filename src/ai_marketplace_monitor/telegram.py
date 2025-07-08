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

    def split_message(self: "TelegramNotificationConfig", title: str, message: str) -> List[str]:
        """Split message into chunks that respect Telegram's 4096 character limit.

        This method splits long messages into multiple chunks while:
        - Respecting word boundaries when possible
        - Preserving formatting for the selected message format
        - Handling edge cases like single words longer than the limit
        - Adding chunk indicators (1/3, 2/3, etc.) for multi-chunk messages

        Args:
            title: The message title
            message: The message content

        Returns:
            List of message chunks, each under 4096 characters

        Raises:
            ValueError: If title or message are empty/invalid
        """
        # Validate inputs
        if not isinstance(title, str) or not title or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(message, str) or not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        title = title.strip()
        message = message.strip()

        # Telegram's message limit
        max_message_length = 4096

        # Create the full message with title
        full_message = f"{title}\n\n{message}"

        # If the complete message fits within the limit, return as single chunk
        if len(full_message) <= max_message_length:
            return [full_message]

        # Message needs to be split - first pass to create raw chunks
        raw_chunks = []
        remaining_message = message
        is_first_chunk = True

        while remaining_message:
            # Reserve space for chunk indicators (e.g., "(1/3) ")
            chunk_indicator_space = 10  # Reserve space for indicators like "(99/99) "

            # Calculate available space for this chunk
            if is_first_chunk:
                # First chunk includes title
                available_space = (
                    max_message_length - len(title) - 2 - chunk_indicator_space
                )  # -2 for "\n\n"
                chunk_prefix = f"{title}\n\n"
            else:
                # Subsequent chunks don't include title
                available_space = max_message_length - chunk_indicator_space
                chunk_prefix = ""

            # If remaining message fits in available space, add it and we're done
            if len(remaining_message) <= available_space:
                raw_chunks.append(chunk_prefix + remaining_message)
                break

            # Find the best split point within available space
            split_point = self._find_split_point(remaining_message, available_space)

            # Extract the chunk content
            chunk_content = remaining_message[:split_point]
            raw_chunks.append(chunk_prefix + chunk_content)

            # Update remaining message
            remaining_message = remaining_message[split_point:].lstrip()
            is_first_chunk = False

        # Second pass: add chunk indicators if we have multiple chunks
        if len(raw_chunks) == 1:
            return raw_chunks

        # Add chunk indicators to each chunk
        final_chunks = []
        total_chunks = len(raw_chunks)

        for i, chunk in enumerate(raw_chunks):
            chunk_number = i + 1
            indicator = f"({chunk_number}/{total_chunks}) "

            # Add indicator at the beginning of each chunk
            final_chunks.append(indicator + chunk)

        return final_chunks

    def _find_split_point(self: "TelegramNotificationConfig", text: str, max_length: int) -> int:
        """Find the best point to split text while respecting word boundaries.

        Args:
            text: The text to split
            max_length: Maximum length for this chunk

        Returns:
            Index where to split the text
        """
        if len(text) <= max_length:
            return len(text)

        # If we have a single word longer than max_length, we have to split it
        if " " not in text[:max_length] and "\n" not in text[:max_length]:
            # Split at character boundary with continuation indicator
            return max_length - 10  # Leave some space for continuation indicator

        # Find the last space or newline within the limit
        split_candidates = []

        # Look for newlines (paragraph breaks are ideal split points)
        for i in range(max_length - 1, -1, -1):
            if text[i] == "\n":
                # Prefer splitting at paragraph breaks
                if i > 0 and text[i - 1] == "\n":
                    split_candidates.append((i + 1, 3))  # High priority
                else:
                    split_candidates.append((i + 1, 2))  # Medium priority

        # Look for spaces (word boundaries)
        for i in range(max_length - 1, -1, -1):
            if text[i] == " ":
                split_candidates.append((i + 1, 1))  # Lower priority
            elif text[i] in ".,!?;:":
                split_candidates.append((i + 1, 2))  # Medium priority (after punctuation)

        # Sort by priority (highest first) and then by position (latest first)
        split_candidates.sort(key=lambda x: (-x[1], -x[0]))

        if split_candidates:
            return split_candidates[0][0]

        # Fallback: split at max_length (shouldn't happen with the logic above)
        return max_length

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
        """Send message via Telegram Bot API with automatic message splitting and retry logic"""
        import asyncio
        import http.client
        import socket
        import ssl
        import time

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

        # Retry configuration
        max_retries = 5
        base_delay = 0.1

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
        async def _send_telegram_message(text: str, parse_mode: str | None):
            bot = telegram.Bot(token=self.telegram_bot_token)
            await bot.send_message(
                chat_id=self.telegram_chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            return True

        # Try primary format first
        formats_to_try = [self.message_format, *fallback_formats]

        for attempt_format in formats_to_try:
            try:
                # Split message into chunks first (using original unescaped message)
                chunks = self.split_message(title, message)

                # Send all chunks for this format with retry logic
                for chunk_text in chunks:
                    # Apply format-specific escaping to each chunk
                    if attempt_format == "plain_text":
                        current_msg_text, current_msg_parse_mode = chunk_text, None
                    elif attempt_format == "markdown":
                        current_msg_text, current_msg_parse_mode = chunk_text, "Markdown"
                    elif attempt_format == "markdownv2":
                        # Apply MarkdownV2 escaping to this chunk
                        current_msg_text = self._escape_markdownv2(chunk_text)
                        current_msg_parse_mode = "MarkdownV2"
                    elif attempt_format == "html":
                        # Apply HTML escaping to this chunk
                        current_msg_text = self._escape_html(chunk_text)
                        current_msg_parse_mode = "HTML"
                    else:
                        # Unknown format, fallback to plain text
                        current_msg_text, current_msg_parse_mode = chunk_text, None

                    # Retry logic for each chunk
                    for retry_attempt in range(max_retries):
                        try:
                            # Execute async function synchronously
                            result = asyncio.run(
                                _send_telegram_message(current_msg_text, current_msg_parse_mode)
                            )

                            if result:
                                break  # Success, move to next chunk
                            else:
                                return False

                        except telegram.error.RetryAfter as e:
                            if retry_attempt < max_retries - 1:
                                sleep_time = e.retry_after
                                if logger:
                                    logger.info(f"Rate limited, retrying in {sleep_time}s")
                                time.sleep(sleep_time)
                            else:
                                if logger:
                                    logger.error(f"Max retries exceeded for rate limit: {e}")
                                return False

                        except (
                            telegram.error.NetworkError,
                            ConnectionError,
                            TimeoutError,
                            asyncio.TimeoutError,
                            OSError,
                            socket.error,
                            socket.gaierror,
                            ssl.SSLError,
                            http.client.HTTPException,
                        ) as e:
                            if retry_attempt < max_retries - 1:
                                sleep_time = base_delay * (2**retry_attempt)
                                if logger:
                                    logger.warning(
                                        f"Network error, retrying in {sleep_time}s: {e}"
                                    )
                                time.sleep(sleep_time)
                            else:
                                if logger:
                                    logger.error(f"Max retries exceeded for network error: {e}")
                                return False

                        except (
                            telegram.error.BadRequest,
                            telegram.error.Forbidden,
                            telegram.error.InvalidToken,
                        ) as e:
                            # Non-retryable errors - check if format-related first
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

                            if is_format_error:
                                # Format error - let outer loop handle fallback
                                raise e
                            else:
                                # Non-retryable non-format error
                                if logger:
                                    logger.error(f"Non-retryable error: {e}")
                                return False

                        except Exception as e:
                            # Check if this might be a format-related error that should trigger fallback
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

                            if is_format_error:
                                # Format error - let outer loop handle fallback
                                raise e

                            # Other errors - retry with exponential backoff
                            if retry_attempt < max_retries - 1:
                                sleep_time = base_delay * (2**retry_attempt)
                                if logger:
                                    logger.warning(
                                        f"Unexpected error, retrying in {sleep_time}s: {e}"
                                    )
                                time.sleep(sleep_time)
                            else:
                                if logger:
                                    logger.error(f"Max retries exceeded for unexpected error: {e}")
                                return False

                # If we get here, all chunks were sent successfully
                if logger and attempt_format != self.message_format:
                    logger.warning(
                        f"Telegram message sent using fallback format '{attempt_format}' instead of '{self.message_format}'"
                    )

                return True

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
