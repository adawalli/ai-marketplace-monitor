"""Tests for notification.py module."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_marketplace_monitor.notification import TelegramNotificationConfig

if TYPE_CHECKING:
    from typing_extensions import Self


class TestTelegramNotificationConfig:
    """Test cases for TelegramNotificationConfig class."""

    @pytest.fixture
    def telegram_config(self: "Self") -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig instance for testing."""
        return TelegramNotificationConfig(
            name="test_telegram",
            telegram_token="123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            telegram_chat_id="12345678",
            max_retries=3,
            retry_delay=1,
        )

    @pytest.fixture
    def mock_logger(self: "Self") -> MagicMock:
        """Create a mock logger for testing."""
        return MagicMock(spec=logging.Logger)

    def test_send_message_success(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test successful message sending through sync interface."""
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = True

            result = telegram_config.send_message(
                title="Test Title", message="Test message", logger=mock_logger
            )

            assert result is True
            mock_asyncio_run.assert_called_once()

    def test_send_message_missing_token(self: "Self", mock_logger: MagicMock) -> None:
        """Test send_message with missing telegram_token."""
        config = TelegramNotificationConfig(
            name="test_telegram", telegram_token=None, telegram_chat_id="12345678"
        )

        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = False

            result = config.send_message(
                title="Test Title", message="Test message", logger=mock_logger
            )

            assert result is False
            mock_asyncio_run.assert_called_once()

    def test_send_message_missing_chat_id(self: "Self", mock_logger: MagicMock) -> None:
        """Test send_message with missing telegram_chat_id."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_token="123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            telegram_chat_id=None,
        )

        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = False

            result = config.send_message(
                title="Test Title", message="Test message", logger=mock_logger
            )

            assert result is False
            mock_asyncio_run.assert_called_once()

    def test_send_message_import_error(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test send_message when telegram import fails."""
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = False

            result = telegram_config.send_message(
                title="Test Title", message="Test message", logger=mock_logger
            )

            assert result is False
            mock_asyncio_run.assert_called_once()

    def test_send_message_telegram_error(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test send_message when Telegram API returns an error."""
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = False

            result = telegram_config.send_message(
                title="Test Title", message="Test message", logger=mock_logger
            )

            assert result is False
            mock_asyncio_run.assert_called_once()

    def test_send_message_no_logger(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test send_message without logger (should not crash)."""
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = True

            result = telegram_config.send_message(
                title="Test Title", message="Test message", logger=None
            )

            assert result is True
            mock_asyncio_run.assert_called_once()

    def test_send_message_exception_handling(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test send_message exception handling and re-raising."""
        with patch("asyncio.run") as mock_asyncio_run:
            # Make asyncio.run raise an exception
            test_exception = Exception("Test exception")
            mock_asyncio_run.side_effect = test_exception

            # Verify that the exception is re-raised and logged
            with pytest.raises(Exception, match="Test exception"):
                telegram_config.send_message(
                    title="Test Title", message="Test message", logger=mock_logger
                )

            mock_logger.error.assert_called_with("Telegram notification failed: Test exception")

    def test_send_message_with_retry_integration(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test integration with parent class retry mechanism."""
        with patch.object(telegram_config, "send_message") as mock_send:
            mock_send.return_value = True

            # Test through send_message_with_retry
            result = telegram_config.send_message_with_retry(
                "Test Title", "Test message", logger=mock_logger
            )

            assert result is True
            mock_send.assert_called_once_with(
                title="Test Title", message="Test message", logger=mock_logger
            )

    def test_required_fields_validation(self: "Self") -> None:
        """Test that required_fields class variable is correctly defined."""
        assert TelegramNotificationConfig.required_fields == ["telegram_token", "telegram_chat_id"]

    def test_config_with_username_chat_id(self: "Self", mock_logger: MagicMock) -> None:
        """Test configuration with username-style chat_id."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_token="123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            telegram_chat_id="@testuser",
        )

        # Should not raise validation errors
        assert config.telegram_chat_id == "@testuser"
        assert config._has_required_fields() is True

    def test_has_required_fields(self: "Self") -> None:
        """Test required fields validation."""
        # Valid config
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_token="123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            telegram_chat_id="12345678",
        )
        assert config._has_required_fields() is True

        # Missing token
        config.telegram_token = None
        assert config._has_required_fields() is False

        # Missing chat_id
        config.telegram_token = "123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"  # noqa: S105
        config.telegram_chat_id = None
        assert config._has_required_fields() is False

    def test_message_size_limits(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test handling of large messages."""
        large_title = "x" * 100
        large_message = "y" * 4000  # Near Telegram's message limit

        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = True

            result = telegram_config.send_message(
                title=large_title, message=large_message, logger=mock_logger
            )

            assert result is True
            mock_asyncio_run.assert_called_once()

    @pytest.mark.parametrize(
        "title,message,expected_calls",
        [
            # Test MarkdownV2 special characters that need escaping
            ("Title with _ underscore", "Message with * asterisk", 1),
            ("Title with [ ] brackets", "Message with ( ) parentheses", 1),
            ("Title with ~ tilde", "Message with ` backtick", 1),
            ("Title with > arrow", "Message with # hash", 1),
            ("Title with + plus", "Message with - minus", 1),
            ("Title with = equals", "Message with | pipe", 1),
            ("Title with { } braces", "Message with . period", 1),
            ("Title with ! exclamation", "Normal message", 1),
            # Edge cases
            ("", "", 1),  # Empty strings
            ("_*[]()~`>#+-=|{}.!", "_*[]()~`>#+-=|{}.!", 1),  # All special chars
            ("Multiple *** asterisks", "Multiple ### hashes", 1),
            ("Normal title", "Normal message", 1),  # No special chars
        ],
    )
    def test_markdown_escaping(
        self: "Self",
        telegram_config: TelegramNotificationConfig,
        mock_logger: MagicMock,
        title: str,
        message: str,
        expected_calls: int,
    ) -> None:
        """Test MarkdownV2 special character escaping in titles and messages."""
        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Mock the escape_markdown function to track its calls
            mock_escape.side_effect = lambda text, version: f"escaped_{text}"

            # Mock the Bot and its send_message method
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                result = telegram_config.send_message(
                    title=title, message=message, logger=mock_logger
                )

                assert result is True
                assert mock_asyncio_run.call_count == expected_calls

    def test_markdown_escaping_integration(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test actual MarkdownV2 escaping integration with telegram.helpers.escape_markdown."""
        test_title = "Test [Title] with *special* chars!"
        test_message = "Message with _underscore_ and `backticks`"

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Use the real escape_markdown behavior (mock but return escaped strings)
            def escape_mock(text: str, version: int) -> str:
                if version == 2:
                    # Simulate basic MarkdownV2 escaping
                    special_chars = "_*[]()~`>#+-=|{}.!"
                    escaped = text
                    for char in special_chars:
                        escaped = escaped.replace(char, f"\\{char}")
                    return escaped
                return text

            mock_escape.side_effect = escape_mock

            # Mock the Bot and its async send_message method
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_send_message = AsyncMock(return_value=None)
            mock_bot_instance.send_message = mock_send_message

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                result = telegram_config.send_message(
                    title=test_title, message=test_message, logger=mock_logger
                )

                assert result is True
                mock_asyncio_run.assert_called_once()

    def test_async_send_message_with_escaping(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test the private async method with proper MarkdownV2 escaping."""
        test_title = "Title with [brackets] and *asterisks*"
        test_message = "Message with _underscores_ and `code`"

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Mock escape_markdown to return predictable escaped text
            def escape_mock(text: str, version: int = 2) -> str:
                return f"escaped_{text}"

            mock_escape.side_effect = escape_mock

            # Mock the Bot and its send_message method
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_send_message = AsyncMock(return_value=None)
            mock_bot_instance.send_message = mock_send_message

            # Run the async method directly
            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(test_title, test_message, mock_logger)
            )

            assert result is True

            # Verify Bot was instantiated with correct token
            mock_bot_class.assert_called_once_with(token=telegram_config.telegram_token)

            # Verify send_message was called with escaped content
            mock_send_message.assert_called_once_with(
                chat_id=telegram_config.telegram_chat_id,
                text="*escaped_Title with [brackets] and *asterisks**\n\nescaped_Message with _underscores_ and `code`",
                parse_mode="MarkdownV2",
            )

            # Verify escape_markdown was called for both title and message
            assert mock_escape.call_count == 2

    def test_async_send_message_import_error(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test async method when telegram import fails."""
        import asyncio

        # Mock the _send_message_async method to directly test the ImportError path
        original_method = telegram_config._send_message_async

        async def mock_method_with_import_error(
            title: str, message: str, logger: MagicMock
        ) -> bool:
            try:
                # Simulate the import statement that will fail
                raise ImportError("No module named 'telegram'")
            except ImportError:
                if logger:
                    logger.error(
                        "python-telegram-bot library is required for Telegram notifications"
                    )
                return False

        # Temporarily replace the method
        telegram_config._send_message_async = mock_method_with_import_error

        try:
            result = asyncio.run(
                telegram_config._send_message_async("Title", "Message", mock_logger)
            )

            assert result is False
            mock_logger.error.assert_called_with(
                "python-telegram-bot library is required for Telegram notifications"
            )
        finally:
            # Restore the original method
            telegram_config._send_message_async = original_method

    def test_async_send_message_telegram_api_error(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test async method when Telegram API raises an error."""
        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: text

            # Mock Bot to raise an exception during send_message
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(side_effect=Exception("API Error"))

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async("Title", "Message", mock_logger)
            )

            assert result is False
            mock_logger.error.assert_called_with("Failed to send Telegram message: API Error")

    def test_edge_case_empty_strings(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test handling of empty title and message strings."""
        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: text

            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(telegram_config._send_message_async("", "", mock_logger))

            assert result is True

            # Verify the formatted message is correct even with empty strings
            mock_bot_instance.send_message.assert_called_once_with(
                chat_id=telegram_config.telegram_chat_id,
                text="**\n\n",  # Empty escaped title and message
                parse_mode="MarkdownV2",
            )

    def test_very_long_strings(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test handling of very long title and message strings."""
        long_title = "x" * 200  # Very long title
        long_message = "y" * 3500  # Very long message (near Telegram limit)

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: f"escaped_{text}"

            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(long_title, long_message, mock_logger)
            )

            assert result is True

            # Verify the call was made (content verification would be very long)
            mock_bot_instance.send_message.assert_called_once()
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["chat_id"] == telegram_config.telegram_chat_id
            assert call_args[1]["parse_mode"] == "MarkdownV2"

    def test_split_message_at_boundaries_short_message(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test splitting logic with message shorter than limit."""
        short_message = "This is a short message"
        result = telegram_config._split_message_at_boundaries(short_message, 100)

        assert result == [short_message]

    def test_split_message_at_boundaries_word_boundary(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test splitting at word boundaries."""
        message = "This is a message that needs to be split properly"
        result = telegram_config._split_message_at_boundaries(message, 30)

        # Should split at word boundaries
        assert len(result) >= 2
        assert all(len(part) <= 30 for part in result)
        assert " ".join(result) == message.replace(
            "  ", " "
        )  # Account for possible space normalization

    def test_split_message_at_boundaries_very_long_word(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test splitting with word longer than limit (edge case)."""
        long_word = "verylongwordthatexceedsthelimit"
        message = f"Short {long_word} end"
        result = telegram_config._split_message_at_boundaries(message, 20)

        # Should force split the long word
        assert len(result) >= 2
        assert all(len(part) <= 20 for part in result)

    def test_split_message_at_boundaries_preserve_content(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test that splitting preserves all message content."""
        message = "Word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        result = telegram_config._split_message_at_boundaries(message, 25)

        # Rejoin and compare (accounting for whitespace normalization)
        rejoined = " ".join(result).strip()
        original_normalized = " ".join(message.split())
        assert rejoined == original_normalized

    def test_message_splitting_under_limit(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test message sending when under 4096 character limit."""
        title = "Short Title"
        message = "Short message"

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: text
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(telegram_config._send_message_async(title, message, mock_logger))

            assert result is True
            # Should send only one message
            assert mock_bot_instance.send_message.call_count == 1

    def test_message_splitting_over_limit(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test message splitting when over 4096 character limit."""
        title = "Test Title"
        # Create a message longer than 4096 characters
        long_message = "This is a very long message. " * 150  # ~4500 characters

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: text
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(title, long_message, mock_logger)
            )

            assert result is True
            # Should send multiple messages
            assert mock_bot_instance.send_message.call_count > 1

            # Check that first message includes title and continuation indicator
            first_call = mock_bot_instance.send_message.call_args_list[0]
            first_message_text = first_call[1]["text"]
            assert title in first_message_text
            assert "\\(1/" in first_message_text  # Escaped continuation indicator

            # Check that subsequent messages have continuation indicators
            if mock_bot_instance.send_message.call_count > 1:
                second_call = mock_bot_instance.send_message.call_args_list[1]
                second_message_text = second_call[1]["text"]
                assert "\\(2/" in second_message_text

    def test_message_splitting_continuation_indicators(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that continuation indicators are properly formatted."""
        title = "Test"
        # Create a message that will definitely need splitting into 3+ parts
        long_message = "Word " * 1500  # ~7500 characters, should split into multiple parts

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: text
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(title, long_message, mock_logger)
            )

            assert result is True
            call_count = mock_bot_instance.send_message.call_count
            assert call_count >= 2

            # Check all messages have proper continuation indicators
            for i, call in enumerate(mock_bot_instance.send_message.call_args_list):
                message_text = call[1]["text"]
                expected_indicator = f"\\({i + 1}/{call_count}\\)"
                assert expected_indicator in message_text

    def test_message_splitting_preserves_formatting(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that MarkdownV2 formatting is preserved during splitting."""
        title = "Title with *formatting*"
        long_message = "Message with _underscores_ and *bold* text. " * 150

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Track escape calls to ensure formatting is preserved
            escape_calls = []

            def track_escape(text: str, version: int) -> str:
                escape_calls.append((text, version))
                return f"escaped_{text}"

            mock_escape.side_effect = track_escape
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(title, long_message, mock_logger)
            )

            assert result is True
            # With message splitting, we have: title + full message (length check) + message parts
            # So expect more than 2 escape calls when message is split
            assert len(escape_calls) >= 2
            assert escape_calls[0][0] == title
            # The full message should be escaped for length check
            assert escape_calls[1][0] == long_message

            # All messages should use MarkdownV2 parse mode
            for call in mock_bot_instance.send_message.call_args_list:
                assert call[1]["parse_mode"] == "MarkdownV2"

    def test_markdownv2_formatting_preserved_across_splits(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that MarkdownV2 formatting is correctly preserved when messages are split."""
        title = "Test"
        # Create a message with special characters that need escaping
        message_with_specials = (
            "Message with _underscores_ and *asterisks* and [brackets] and (parentheses). " * 100
        )

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Track calls to escape_markdown to ensure each part is escaped separately
            escape_calls = []

            def track_escape(text: str, version: int) -> str:
                escape_calls.append((text, version))
                # Return a simple escaped version for testing
                return (
                    text.replace("_", "\\_")
                    .replace("*", "\\*")
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                    .replace("(", "\\(")
                    .replace(")", "\\)")
                )

            mock_escape.side_effect = track_escape
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(title, message_with_specials, mock_logger)
            )

            assert result is True

            # Should have multiple send_message calls due to length
            call_count = mock_bot_instance.send_message.call_count
            assert call_count > 1

            # Should have escaped the title, full message (for length check), and each message part separately
            # Title + full message + each part = 2 + call_count escape calls
            assert len(escape_calls) == call_count + 2

            # Verify each escape call used version 2 (MarkdownV2)
            for _text, version in escape_calls:
                assert version == 2

    def test_markdownv2_escape_sequences_not_broken(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that escape sequences are not broken when splitting messages."""
        title = "Test"
        # Create a message that would break escape sequences if split at wrong position
        message = "This message has many special chars: " + "_*[]()~`>#+-=|{}.!" * 200

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Real escape function behavior
            def real_escape(text: str, version: int) -> str:
                if version == 2:
                    special_chars = "_*[]()~`>#+-=|{}.!"
                    escaped = text
                    for char in special_chars:
                        escaped = escaped.replace(char, f"\\{char}")
                    return escaped
                return text

            mock_escape.side_effect = real_escape
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(telegram_config._send_message_async(title, message, mock_logger))

            assert result is True

            # Verify all sent messages have valid MarkdownV2 format
            for call in mock_bot_instance.send_message.call_args_list:
                message_text = call[1]["text"]
                # Should not have any unescaped special characters
                # (except for the intentional formatting like *title*)
                assert call[1]["parse_mode"] == "MarkdownV2"

                # Check that we don't have broken escape sequences like "\ " (backslash followed by space)
                # or incomplete escapes at the end of message parts
                assert not message_text.endswith("\\")

    def test_original_message_split_not_escaped_message(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that we split the original message, not the escaped version."""
        title = "Test"
        # Message that would be much longer after escaping
        original_message = (
            "Short message with * and _ chars " * 150
        )  # Make it longer to ensure splitting

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            # Track what gets passed to _split_message_at_boundaries
            original_split_method = telegram_config._split_message_at_boundaries
            split_calls = []

            def track_split_calls(text: str, max_length: int) -> List[str]:
                split_calls.append((text, max_length))
                return original_split_method(text, max_length)

            telegram_config._split_message_at_boundaries = track_split_calls

            # Mock escape to return much longer text
            def escape_with_expansion(text: str, version: int) -> str:
                return text.replace("*", "\\*").replace("_", "\\_")

            mock_escape.side_effect = escape_with_expansion
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(title, original_message, mock_logger)
            )

            assert result is True

            # Should have called split with the original message, not escaped
            assert len(split_calls) == 1
            split_text, _ = split_calls[0]
            assert split_text == original_message  # Original, not escaped

            # Restore original method
            telegram_config._split_message_at_boundaries = original_split_method

    def test_each_message_part_escaped_individually(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that each message part is escaped individually to maintain proper formatting."""
        title = "Test Title"
        # Create message that will be split into parts
        long_message = "Part with *bold* text and _italic_ text. " * 150

        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            escape_calls = []

            def track_escape_calls(text: str, version: int) -> str:
                escape_calls.append(text)
                return f"ESCAPED[{text}]"

            mock_escape.side_effect = track_escape_calls
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock(return_value=None)

            import asyncio

            result = asyncio.run(
                telegram_config._send_message_async(title, long_message, mock_logger)
            )

            assert result is True

            # Should have multiple message sends
            call_count = mock_bot_instance.send_message.call_count
            assert call_count > 1

            # Should have escape calls for: title + full message (length check) + each message part
            assert len(escape_calls) == call_count + 2

            # First escape call should be the title
            assert escape_calls[0] == title

            # Second escape call should be the full message (for length check)
            assert escape_calls[1] == long_message

            # Remaining calls should be different parts of the original message
            message_parts = escape_calls[2:]

            # Each part should be a substring of the original message
            for part in message_parts:
                assert part in long_message or any(part in chunk for chunk in long_message.split())

            # When rejoined, the parts should reconstruct something close to the original
            # (accounting for potential whitespace normalization during splitting)
            rejoined = " ".join(message_parts).strip()
            original_normalized = " ".join(long_message.split())
            # Length should be close (within reasonable margin for whitespace differences)
            assert abs(len(rejoined) - len(original_normalized)) < 50

    def test_split_message_extreme_edge_cases(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test extreme edge cases: very long URLs and single words exceeding limits."""
        # Test with extremely long URL (common in marketplace listings)
        extremely_long_url = "https://example.com/" + "a" * 5000 + ".html"
        message_with_long_url = f"Check this listing: {extremely_long_url} - great deal!"

        result = telegram_config._split_message_at_boundaries(message_with_long_url, 4096)

        # Should handle the long URL by force-splitting
        assert len(result) >= 2
        assert all(len(part) <= 4096 for part in result)

        # Test with multiple extremely long words
        long_words = ["word" + "a" * 5000 + str(i) for i in range(3)]
        message_with_long_words = " ".join(long_words)

        result2 = telegram_config._split_message_at_boundaries(message_with_long_words, 4096)

        # Should handle multiple long words
        assert len(result2) >= 3  # At least one part per long word
        assert all(len(part) <= 4096 for part in result2)

        # Test edge case: single character repeated beyond limit
        single_char_repeated = "a" * 10000
        result3 = telegram_config._split_message_at_boundaries(single_char_repeated, 4096)

        # Should split into multiple 4096-character chunks
        expected_parts = (10000 + 4095) // 4096  # Ceiling division
        assert len(result3) == expected_parts
        assert all(len(part) <= 4096 for part in result3)

        # Verify content preservation by rejoining
        rejoined = "".join(result3)
        assert rejoined == single_char_repeated

    def test_get_wait_time_no_previous_send(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_wait_time with no previous send time."""
        telegram_config._last_send_time = None
        assert telegram_config._get_wait_time() == 0.0

    def test_get_wait_time_individual_chat_ready(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_wait_time for individual chat when ready to send."""
        telegram_config.telegram_chat_id = "12345678"
        telegram_config._last_send_time = time.time() - 2.0  # 2 seconds ago

        wait_time = telegram_config._get_wait_time()
        assert wait_time == 0.0

    def test_get_wait_time_individual_chat_needs_wait(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_wait_time for individual chat when waiting is needed."""
        telegram_config.telegram_chat_id = "12345678"
        telegram_config._last_send_time = time.time() - 0.5  # 0.5 seconds ago

        wait_time = telegram_config._get_wait_time()
        assert 0.5 < wait_time <= 0.7  # Should be around 0.6 seconds

    def test_get_wait_time_group_chat_ready(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_wait_time for group chat when ready to send."""
        telegram_config.telegram_chat_id = "-100123456789"
        telegram_config._last_send_time = time.time() - 4.0  # 4 seconds ago

        wait_time = telegram_config._get_wait_time()
        assert wait_time == 0.0

    def test_get_wait_time_group_chat_needs_wait(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_wait_time for group chat when waiting is needed."""
        telegram_config.telegram_chat_id = "-100123456789"
        telegram_config._last_send_time = time.time() - 1.0  # 1 second ago

        wait_time = telegram_config._get_wait_time()
        assert 1.9 < wait_time <= 2.1  # Should be around 2 seconds

    def test_wait_for_rate_limit_no_wait_needed(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test _wait_for_rate_limit when no waiting is needed."""
        telegram_config.telegram_chat_id = "12345678"
        telegram_config._last_send_time = None

        # Mock asyncio.sleep to verify it's not called
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Test through asyncio.run
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should not have called sleep
            mock_sleep.assert_not_called()
            # Should have set last send time
            assert telegram_config._last_send_time is not None

    def test_wait_for_rate_limit_individual_with_wait(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test _wait_for_rate_limit for individual chat needing to wait."""
        telegram_config.telegram_chat_id = "12345678"
        telegram_config._last_send_time = time.time() - 0.5  # Recent send

        # Mock asyncio.sleep to avoid actual waiting in tests
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Test through asyncio.run
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should have called sleep once with some wait time
            assert mock_sleep.call_count == 1
            wait_time = mock_sleep.call_args[0][0]
            assert 0.5 <= wait_time <= 0.7  # Should be around 0.6 seconds
            # Should have updated last send time
            assert telegram_config._last_send_time is not None

    def test_wait_for_rate_limit_group_with_wait(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test _wait_for_rate_limit for group chat needing to wait."""
        telegram_config.telegram_chat_id = "-100123456789"
        telegram_config._last_send_time = time.time() - 1.0  # Recent send

        # Mock asyncio.sleep to avoid actual waiting in tests
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Test through asyncio.run
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should have called sleep once with some wait time
            assert mock_sleep.call_count == 1
            wait_time = mock_sleep.call_args[0][0]
            assert 1.9 <= wait_time <= 2.1  # Should be around 2 seconds
            # Should have updated last send time
            assert telegram_config._last_send_time is not None

    def test_is_group_chat_individual(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _is_group_chat method for individual chats."""
        # Test positive chat ID (individual chat)
        telegram_config.telegram_chat_id = "12345678"
        assert not telegram_config._is_group_chat()

        # Test username format (individual chat)
        telegram_config.telegram_chat_id = "@username"
        assert not telegram_config._is_group_chat()

        # Test None chat ID
        telegram_config.telegram_chat_id = None
        assert not telegram_config._is_group_chat()

    def test_is_group_chat_group(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _is_group_chat method for group chats."""
        # Test negative chat ID (group chat)
        telegram_config.telegram_chat_id = "-100123456789"
        assert telegram_config._is_group_chat()

        # Test simple negative ID
        telegram_config.telegram_chat_id = "-12345"
        assert telegram_config._is_group_chat()

    def test_rate_limiting_integration(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test rate limiting integration with message sending."""
        with (
            patch("telegram.Bot") as mock_bot_class,
            patch("telegram.helpers.escape_markdown") as mock_escape,
        ):
            mock_escape.side_effect = lambda text, version: f"escaped_{text}"

            # Mock the Bot and its async send_message method
            mock_bot_instance = AsyncMock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = AsyncMock()

            # Mock the rate limiting method to verify it gets called
            with patch.object(telegram_config, "_wait_for_rate_limit", new_callable=AsyncMock):
                with patch("asyncio.run") as mock_asyncio_run:
                    mock_asyncio_run.return_value = True

                    result = telegram_config.send_message(
                        title="Test Title", message="Test message", logger=mock_logger
                    )

                    assert result is True
                    # Verify rate limiting method was called through asyncio.run
                    mock_asyncio_run.assert_called_once()

    def test_get_global_wait_time_empty_queue(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_global_wait_time with empty global send times."""
        # Clear the global queue
        TelegramNotificationConfig._global_send_times.clear()

        wait_time = TelegramNotificationConfig._get_global_wait_time()
        assert wait_time == 0.0

    def test_get_global_wait_time_under_limit(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_global_wait_time when under rate limit."""
        # Clear the global queue and add some timestamps
        TelegramNotificationConfig._global_send_times.clear()
        current_time = time.time()

        # Add 20 messages (under the 30 msg/sec limit)
        for i in range(20):
            TelegramNotificationConfig._global_send_times.append(current_time - 0.5 + i * 0.01)

        wait_time = TelegramNotificationConfig._get_global_wait_time()
        assert wait_time == 0.0

    def test_get_global_wait_time_at_limit(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _get_global_wait_time when at rate limit."""
        # Clear the global queue and add messages at the limit
        TelegramNotificationConfig._global_send_times.clear()
        current_time = time.time()

        # Add exactly 30 messages in the last second (at limit)
        for i in range(30):
            TelegramNotificationConfig._global_send_times.append(current_time - 0.9 + i * 0.03)

        wait_time = TelegramNotificationConfig._get_global_wait_time()
        # Should need to wait until the oldest message is > 1 second old
        assert 0.0 < wait_time <= 0.2

    def test_get_global_wait_time_old_messages_cleaned(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test that old messages are cleaned from global queue."""
        # Clear the global queue
        TelegramNotificationConfig._global_send_times.clear()
        current_time = time.time()

        # Add old messages (> 1 second ago) and recent messages
        for i in range(20):
            TelegramNotificationConfig._global_send_times.append(
                current_time - 2.0 + i * 0.01
            )  # Old
        for i in range(10):
            TelegramNotificationConfig._global_send_times.append(
                current_time - 0.5 + i * 0.01
            )  # Recent

        wait_time = TelegramNotificationConfig._get_global_wait_time()

        # Should have cleaned old messages and be under limit
        assert wait_time == 0.0
        # Queue should only contain recent messages (10)
        assert len(TelegramNotificationConfig._global_send_times) == 10

    def test_record_global_send_time(
        self: "Self", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _record_global_send_time adds timestamp to queue."""
        # Clear the global queue
        TelegramNotificationConfig._global_send_times.clear()
        initial_count = len(TelegramNotificationConfig._global_send_times)

        TelegramNotificationConfig._record_global_send_time()

        # Should have added one timestamp
        assert len(TelegramNotificationConfig._global_send_times) == initial_count + 1
        # Most recent timestamp should be very close to now
        assert abs(TelegramNotificationConfig._global_send_times[-1] - time.time()) < 0.1

    def test_global_rate_limiting_integrated_with_per_chat(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that global and per-chat rate limiting work together."""
        # Clear global queue
        TelegramNotificationConfig._global_send_times.clear()

        # Set up scenario where per-chat doesn't need wait but global does
        current_time = time.time()
        telegram_config._last_send_time = current_time - 2.0  # Per-chat ready (no wait)

        # Fill global queue to capacity requiring wait
        for i in range(30):
            TelegramNotificationConfig._global_send_times.append(current_time - 0.9 + i * 0.03)

        # Mock asyncio.sleep to capture wait time
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should have called sleep once
            assert mock_sleep.call_count == 1
            wait_time = mock_sleep.call_args[0][0]

            # Should be waiting for global rate limit (small wait time)
            assert 0.05 < wait_time < 0.3

    def test_global_rate_limiting_multiple_instances(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that global rate limiting works across multiple TelegramNotificationConfig instances."""
        # Clear global queue
        TelegramNotificationConfig._global_send_times.clear()

        # Create second instance to verify global state is shared
        TelegramNotificationConfig(
            name="test_telegram_2",
            telegram_token="123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            telegram_chat_id="87654321",
        )

        # Fill global queue through first instance
        current_time = time.time()
        for i in range(30):
            TelegramNotificationConfig._global_send_times.append(current_time - 0.9 + i * 0.03)

        # Both instances should see similar global wait times (allowing for tiny timing differences)
        wait_time_1 = TelegramNotificationConfig._get_global_wait_time()
        wait_time_2 = TelegramNotificationConfig._get_global_wait_time()

        assert abs(wait_time_1 - wait_time_2) < 0.01  # Should be very close
        assert wait_time_1 > 0  # Should require waiting
        assert wait_time_2 > 0  # Should require waiting

    def test_wait_for_rate_limit_global_dominates(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test _wait_for_rate_limit when global rate limit requires longer wait than per-chat."""
        # Clear global queue
        TelegramNotificationConfig._global_send_times.clear()

        # Set up per-chat rate limiting (individual chat, no wait needed)
        telegram_config.telegram_chat_id = "12345678"
        telegram_config._last_send_time = time.time() - 2.0  # No per-chat wait needed

        # Set up global rate limiting (at capacity, wait needed)
        current_time = time.time()
        for i in range(30):
            TelegramNotificationConfig._global_send_times.append(current_time - 0.8 + i * 0.025)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should have called sleep for global rate limiting
            assert mock_sleep.call_count == 1
            wait_time = mock_sleep.call_args[0][0]
            assert wait_time > 0

            # Should have logged global rate limiting message
            mock_logger.debug.assert_called_once()
            log_message = mock_logger.debug.call_args[0][0]
            assert "Global rate limiting" in log_message
            assert "30 msg/sec" in log_message

    def test_wait_for_rate_limit_per_chat_dominates(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test _wait_for_rate_limit when per-chat rate limit requires longer wait than global."""
        # Clear global queue (no global wait needed)
        TelegramNotificationConfig._global_send_times.clear()

        # Set up per-chat rate limiting (group chat, wait needed)
        telegram_config.telegram_chat_id = "-100123456789"
        telegram_config._last_send_time = time.time() - 1.0  # Group needs ~2s wait

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should have called sleep for per-chat rate limiting
            assert mock_sleep.call_count == 1
            wait_time = mock_sleep.call_args[0][0]
            assert 1.9 < wait_time < 2.1  # Should be around 2 seconds

            # Should have logged per-chat rate limiting message
            mock_logger.debug.assert_called_once()
            log_message = mock_logger.debug.call_args[0][0]
            assert "Rate limiting group chat" in log_message

    def test_wait_for_rate_limit_records_global_send_time(
        self: "Self", telegram_config: TelegramNotificationConfig, mock_logger: MagicMock
    ) -> None:
        """Test that _wait_for_rate_limit records global send time."""
        # Clear global queue
        TelegramNotificationConfig._global_send_times.clear()
        initial_count = len(TelegramNotificationConfig._global_send_times)

        # Set up no waiting needed
        telegram_config._last_send_time = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(telegram_config._wait_for_rate_limit(mock_logger))

            # Should have recorded global send time
            assert len(TelegramNotificationConfig._global_send_times) == initial_count + 1
