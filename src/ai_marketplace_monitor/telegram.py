import html
from dataclasses import dataclass
from logging import Logger
from typing import ClassVar, List

import requests

from .notification import PushNotificationConfig
from .utils import hilight


@dataclass
class TelegramNotificationConfig(PushNotificationConfig):
    notify_method = "telegram"
    required_fields: ClassVar[List[str]] = ["telegram_bot_token", "telegram_chat_id"]

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    def handle_telegram_bot_token(self: "TelegramNotificationConfig") -> None:
        if self.telegram_bot_token is None:
            return
        if not isinstance(self.telegram_bot_token, str) or not self.telegram_bot_token:
            raise ValueError("An non-empty telegram_bot_token is needed.")
        self.telegram_bot_token = self.telegram_bot_token.strip()

    def handle_telegram_chat_id(self: "TelegramNotificationConfig") -> None:
        if self.telegram_chat_id is None:
            return
        if not isinstance(self.telegram_chat_id, str) or not self.telegram_chat_id:
            raise ValueError("An non-empty telegram_chat_id is needed.")
        self.telegram_chat_id = self.telegram_chat_id.strip()

    def handle_message_format(self: "TelegramNotificationConfig") -> None:
        # Store original value to check if it was None before parent processing
        was_none = self.message_format is None
        super().handle_message_format()
        # If it was originally None, override the parent's "plain_text" default with "markdown"
        if was_none:
            self.message_format = "markdown"

    def send_message(
        self: "TelegramNotificationConfig",
        title: str,
        message: str,
        logger: Logger | None = None,
    ) -> bool:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            if logger:
                logger.error(
                    "telegram_bot_token and telegram_chat_id must be set before calling send_message()"
                )
            return False

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"

        # Escape functions for safe formatting
        def md_escape(text: str) -> str:
            """Escape special characters for MarkdownV2"""
            # Characters that need escaping in MarkdownV2: _ * [ ] ( ) ~ ` > # + - = | { } . ! \
            special_chars = "_*[]()~`>#+\\-=|{}.!\\"
            escape_map = str.maketrans({char: f"\\{char}" for char in special_chars})
            return text.translate(escape_map)

        def html_escape(text: str) -> str:
            """Escape special characters for HTML"""
            return html.escape(text)

        # Helper function to format text based on message_format
        def format_bold(text: str) -> str:
            if self.message_format == "html":
                return f"<b>{html_escape(text)}</b>"
            elif self.message_format == "markdown":
                return f"*{md_escape(text)}*"
            else:
                return text

        def format_italic_link(text: str, url: str) -> str:
            if self.message_format == "html":
                return f'<i><a href="{html_escape(url)}">{html_escape(text)}</a></i>'
            elif self.message_format == "markdown":
                return f"_[{md_escape(text)}]({url})_"
            else:
                return f"{text}: {url}"

        # Calculate maximum overhead for title and signature
        max_title_part = f"{format_bold(title)} (999/999)\n\n"  # Worst case part numbering
        signature = f"\n\n{format_italic_link('Sent by AI Marketplace Monitor', 'https://github.com/BoPeng/ai-marketplace-monitor')}"
        max_overhead = len(max_title_part) + len(signature)

        # Telegram's limit is 4096, leave buffer and account for overhead
        max_content_length = 4096 - max_overhead - 50  # Extra buffer for safety

        # Split message if it's too long
        messages = []
        if len(message) <= max_content_length:
            messages.append(message)
        else:
            # Split by '\n\n' which separates listings
            pieces = message.split("\n\n")
            current_msg = ""

            for piece in pieces:
                test_msg = current_msg + "\n\n" + piece if current_msg else piece
                if len(test_msg) <= max_content_length:
                    current_msg = test_msg
                else:
                    if current_msg:
                        messages.append(current_msg)
                        current_msg = piece
                    else:
                        # Single piece is too long, split it further
                        # This is a fallback for extremely long individual listings
                        while len(piece) > max_content_length:
                            split_point = piece.rfind(" ", 0, max_content_length)
                            if split_point == -1:
                                split_point = max_content_length
                            messages.append(piece[:split_point])
                            piece = piece[split_point:].lstrip()
                        current_msg = piece

            if current_msg:
                messages.append(current_msg)

        # Send each message part
        for idx, msg in enumerate(messages):
            title_part = format_bold(title)
            if len(messages) > 1:
                title_part += f" ({idx + 1}/{len(messages)})"

            full_message = f"{title_part}\n\n{msg}"

            # Add signature to the last message
            if idx == len(messages) - 1:
                full_message += f"\n\n{format_italic_link('Sent by AI Marketplace Monitor', 'https://github.com/BoPeng/ai-marketplace-monitor')}"

            # Final safety check - if somehow still too long, truncate
            if len(full_message) > 4096:
                available_space = (
                    4096
                    - len(title_part)
                    - len("\n\n")
                    - (len(signature) if idx == len(messages) - 1 else 0)
                    - 20
                )
                # Ensure available_space is never negative to prevent malformed slicing
                available_space = max(0, available_space)
                msg = msg[:available_space] + "..."
                full_message = f"{title_part}\n\n{msg}"
                if idx == len(messages) - 1:
                    full_message += f"\n\n{format_italic_link('Sent by AI Marketplace Monitor', 'https://github.com/BoPeng/ai-marketplace-monitor')}"

            payload = {
                "chat_id": self.telegram_chat_id,
                "text": full_message,
                "disable_web_page_preview": True,
            }

            # Set parse mode based on message format
            if self.message_format == "markdown":
                payload["parse_mode"] = "MarkdownV2"
            elif self.message_format == "html":
                payload["parse_mode"] = "HTML"

            try:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()

                result = response.json()
                if not result.get("ok", False):
                    if logger:
                        logger.error(
                            f"Telegram API error: {result.get('description', 'Unknown error')}"
                        )
                    return False

            except requests.exceptions.RequestException as e:
                if logger:
                    logger.error(f"Failed to send Telegram message: {e}")
                return False

        if logger:
            logger.info(
                f"""{hilight("[Notify]", "succ")} Sent {self.name} a message with title {hilight(title)}"""
            )
        return True
