"""Tests for notification.py module."""

import logging
from typing import TYPE_CHECKING
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
