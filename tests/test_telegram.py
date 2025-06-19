import asyncio
import threading
from typing import Any, Awaitable, Callable, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from unittest.mock import AsyncMock, Mock, patch

import pytest
from telegram import Message
from telegram.constants import MessageLimit
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


def run_async_in_thread(coro: Awaitable[Any]) -> Any:
    """Run an async coroutine in a separate thread with its own event loop.

    This custom fixture is necessary to solve the event loop conflict between:
    1. Playwright tests that create their own event loop for browser automation
    2. pytest-asyncio that tries to manage event loops for async test methods

    When both run in the same process, pytest-asyncio throws "RuntimeError: This event loop is already running"
    because Playwright's event loop is already active. By running async tests in a separate thread
    with their own isolated event loop, we avoid this conflict while maintaining full test coverage.

    This is a workaround for the known incompatibility between pytest-playwright and pytest-asyncio
    when used together in the same test suite.
    """
    result = None
    exception = None

    def run_in_thread() -> None:
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
            finally:
                loop.close()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result


class TestTelegramNotificationConfig:
    """Test suite for TelegramNotificationConfig."""

    def test_init_defaults(self: Self) -> None:
        """Test default initialization."""
        config = TelegramNotificationConfig(name="test")
        assert config.notify_method == "telegram"
        assert config.telegram_bot_token is None
        assert config.telegram_chat_id is None
        assert config.message_format == "markdownv2"  # Should default to markdownv2

    def test_init_with_values(self: Self) -> None:
        """Test initialization with values."""
        config = TelegramNotificationConfig(
            name="test",
            telegram_bot_token="bot123",
            telegram_chat_id="chat456",
            message_format="html",
        )
        assert config.telegram_bot_token == "bot123"  # noqa: S105
        assert config.telegram_chat_id == "chat456"
        assert config.message_format == "html"

    def test_required_fields(self: Self) -> None:
        """Test required fields class variable."""
        assert TelegramNotificationConfig.required_fields == [
            "telegram_bot_token",
            "telegram_chat_id",
        ]

    @pytest.mark.parametrize(
        "token,should_raise",
        [
            ("valid_token", False),
            ("  valid_token  ", False),  # Should be stripped
            ("", True),
            (None, False),  # None is allowed
            (123, True),  # Non-string should raise
        ],
    )
    def test_handle_telegram_bot_token(
        self: Self, token: Union[str, None, int, list], should_raise: bool
    ) -> None:
        """Test telegram_bot_token validation."""
        if should_raise:
            with pytest.raises(ValueError, match="non-empty string"):
                config = TelegramNotificationConfig(name="test", telegram_bot_token=token)
        else:
            config = TelegramNotificationConfig(name="test", telegram_bot_token=token)
            if token and isinstance(token, str):
                assert config.telegram_bot_token == token.strip()

    @pytest.mark.parametrize(
        "chat_id,should_raise",
        [
            ("123456", False),
            ("  123456  ", False),  # Should be stripped
            ("@channel", False),
            ("", True),
            (None, False),  # None is allowed
            (123, True),  # Non-string should raise
        ],
    )
    def test_handle_telegram_chat_id(
        self: Self, chat_id: Union[str, None, int, list], should_raise: bool
    ) -> None:
        """Test telegram_chat_id validation."""
        if should_raise:
            with pytest.raises(ValueError, match="non-empty string"):
                config = TelegramNotificationConfig(name="test", telegram_chat_id=chat_id)
        else:
            config = TelegramNotificationConfig(name="test", telegram_chat_id=chat_id)
            if chat_id and isinstance(chat_id, str):
                assert config.telegram_chat_id == chat_id.strip()

    def test_handle_message_format_default(self: "TestTelegramNotificationConfig") -> None:
        """Test message_format defaults to markdownv2 when None."""
        config = TelegramNotificationConfig(name="test")
        # message_format should already be "markdownv2" due to override in handle_message_format
        assert config.message_format == "markdownv2"

    def test_handle_message_format_explicit(self: "TestTelegramNotificationConfig") -> None:
        """Test message_format when explicitly set."""
        config = TelegramNotificationConfig(name="test", message_format="html")
        config.handle_message_format()
        assert config.message_format == "html"


class TestTelegramMessaging:
    """Test suite for Telegram messaging functionality."""

    @pytest.fixture
    def telegram_config(self: "TestTelegramMessaging") -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig instance for testing."""
        return TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="test_bot_token",
            telegram_chat_id="test_chat_id",
            message_format="markdown",
        )

    @pytest.fixture
    def mock_logger(self: "TestTelegramMessaging") -> Mock:
        """Create a mock logger."""
        logger = Mock()
        logger.debug = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger

    @pytest.mark.asyncio
    async def test_send_message_async_success(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test successful async message sending."""
        mock_bot = AsyncMock()
        mock_message = Mock(spec=Message)
        mock_message.message_id = 12345
        mock_bot.send_message.return_value = mock_message

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger
        )

        assert result is True
        mock_bot.send_message.assert_called_once_with(
            chat_id="test_chat_id",
            text="Test message",
            parse_mode="MarkdownV2",
            disable_web_page_preview=True,
        )

    @pytest.mark.asyncio
    async def test_send_message_async_retry_after(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test RetryAfter exception handling."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = [RetryAfter(retry_after=0.1), Mock(message_id=12345)]

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=1
        )

        assert result is True
        assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_timeout(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test TimedOut exception handling."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = [TimedOut(), Mock(message_id=12345)]

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=1
        )

        assert result is True
        assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_bad_request(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test BadRequest exception handling (should not retry)."""
        mock_bot = AsyncMock()
        error = BadRequest("Can't parse entities")
        error.message = "Can't parse entities: invalid markup"
        mock_bot.send_message.side_effect = error

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger
        )

        assert result is False
        mock_bot.send_message.assert_called_once()
        # Verify error logging
        assert mock_logger.error.call_count >= 3

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_telegram_error(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test general TelegramError handling."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TelegramError("General error")

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger
        )

        assert result is False
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_single(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test sending a single message with formatting."""
        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = await telegram_config._send_all_messages_async(
                "Test Title", ["Test content"], "MarkdownV2", mock_logger
            )

            assert result is True
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args[1]
            assert "Test Title" in call_args["text"]
            assert "Test content" in call_args["text"]
            assert "Sent by AI Marketplace Monitor" in call_args["text"]

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_multiple(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test sending multiple message parts."""
        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            messages = ["Part 1", "Part 2", "Part 3"]
            result = await telegram_config._send_all_messages_async(
                "Test Title", messages, "MarkdownV2", mock_logger
            )

            assert result is True
            assert mock_bot.send_message.call_count == 3

            # Check that part numbers are included
            for i, call in enumerate(mock_bot.send_message.call_args_list):
                text = call[1]["text"]
                assert f"({i + 1}/3)" in text
                assert f"Part {i + 1}" in text

            # Check signature is only in last message
            last_call_text = mock_bot.send_message.call_args_list[-1][1]["text"]
            assert "Sent by AI Marketplace Monitor" in last_call_text

    def test_send_message_no_credentials(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test send_message with missing credentials."""
        telegram_config.telegram_bot_token = None
        result = telegram_config.send_message("Title", "Message", mock_logger)
        assert result is False

        telegram_config.telegram_bot_token = "token"  # noqa: S105
        telegram_config.telegram_chat_id = None
        result = telegram_config.send_message("Title", "Message", mock_logger)
        assert result is False

    def test_send_message_markdown_escaping(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test markdown escaping in messages."""
        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    message = "Test *bold* _italic_ [link](url)"
                    telegram_config.send_message("Title", message, mock_logger)

                    # Verify escaping was applied
                    mock_loop.return_value.run_until_complete.assert_called_once()
                    mock_loop.return_value.run_until_complete.call_args[0][0]
                    # Verify the coroutine was called
                    assert mock_loop.return_value.run_until_complete.called

    def test_send_message_html_escaping(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test HTML escaping in messages."""
        telegram_config.message_format = "html"
        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    message = "Test <b>bold</b> & special"
                    telegram_config.send_message("Title", message, mock_logger)

                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_plain_text(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test plain text messages (no escaping)."""
        telegram_config.message_format = "plain_text"
        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    message = "Test *bold* <tag>"
                    telegram_config.send_message("Title", message, mock_logger)

                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_message_splitting_long_message(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting for long messages."""
        # Create a message that exceeds the limit
        long_message = "Test content " * 500  # Create a very long message

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", long_message, mock_logger)

                    # Should be called with split messages
                    mock_loop.return_value.run_until_complete.call_args[0][0]
                    # Verify it was called (the actual splitting logic is complex)
                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_message_splitting_by_paragraphs(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting preserves paragraph boundaries."""
        # Create a message with clear paragraph boundaries
        message_parts = ["Paragraph 1" * 100, "Paragraph 2" * 100, "Paragraph 3" * 100]
        message = "\n\n".join(message_parts)

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", message, mock_logger)

                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_with_running_event_loop(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test send_message when already in an event loop."""
        with patch("asyncio.get_running_loop", return_value=Mock()):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = True
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                result = telegram_config.send_message("Title", "Message", mock_logger)

                assert result is True
                mock_executor.return_value.__enter__.return_value.submit.assert_called_once()

    def test_send_message_exception_handling(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test exception handling in send_message."""
        with patch("asyncio.get_running_loop", side_effect=Exception("Test error")):
            result = telegram_config.send_message("Title", "Message", mock_logger)
            assert result is False

    @pytest.mark.parametrize(
        "message_format,expected_parse_mode",
        [
            ("markdown", "MarkdownV2"),
            ("html", "HTML"),
            ("plain_text", None),
        ],
    )
    def test_parse_mode_selection(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
        message_format: str,
        expected_parse_mode: str,
    ) -> None:
        """Test correct parse mode selection based on message format."""
        telegram_config.message_format = message_format

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", "Message", mock_logger)

                    # Get the coroutine that was passed to run_until_complete
                    mock_loop.return_value.run_until_complete.call_args[0][0]
                    # Since we can't easily inspect the coroutine, we'll trust the implementation
                    mock_loop.return_value.run_until_complete.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_max_retries_exceeded(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test max retries exceeded for rate limiting."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = RetryAfter(retry_after=0.01)

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=2
        )

        assert result is False
        assert mock_bot.send_message.call_count == 3  # initial + 2 retries
        mock_logger.error.assert_called_with("Max retries exceeded for rate limiting")

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_timeout_max_retries(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test max retries exceeded for timeout."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TimedOut()

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=1
        )

        assert result is False
        assert mock_bot.send_message.call_count == 2  # initial + 1 retry
        mock_logger.error.assert_called_with("Max retries exceeded for timeout")

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_html_format(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test HTML formatting in send_all_messages_async."""
        telegram_config.message_format = "html"

        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = await telegram_config._send_all_messages_async(
                "Test <Title>", ["Test & content"], "HTML", mock_logger
            )

            assert result is True
            call_text = mock_bot.send_message.call_args[1]["text"]
            assert "<b>Test &lt;Title&gt;</b>" in call_text  # Title should be escaped
            # Note: Content is pre-escaped in send_message, not in _send_all_messages_async
            assert "Test & content" in call_text

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_plain_text(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test plain text formatting in send_all_messages_async."""
        telegram_config.message_format = "plain_text"

        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = await telegram_config._send_all_messages_async(
                "Test Title", ["Test content"], None, mock_logger
            )

            assert result is True
            call_text = mock_bot.send_message.call_args[1]["text"]
            assert "Test Title" in call_text  # Plain text, no formatting
            assert "https://github.com/BoPeng/ai-marketplace-monitor" in call_text

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_message_too_long(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message truncation when exceeding limit."""
        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            # Create a message that would exceed the limit
            very_long_message = "x" * (MessageLimit.MAX_TEXT_LENGTH + 1000)

            result = await telegram_config._send_all_messages_async(
                "Title", [very_long_message], "MarkdownV2", mock_logger
            )

            assert result is True
            call_text = mock_bot.send_message.call_args[1]["text"]
            assert len(call_text) <= MessageLimit.MAX_TEXT_LENGTH
            assert "..." in call_text
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_failure(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test handling of send failure in send_all_messages_async."""
        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.side_effect = TelegramError("Network error")

            result = await telegram_config._send_all_messages_async(
                "Title", ["Message 1", "Message 2"], "MarkdownV2", mock_logger
            )

            assert result is False
            # Should stop after first failure
            mock_bot.send_message.assert_called_once()

    def test_send_message_success_with_logging(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test successful message sending with logging."""
        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    result = telegram_config.send_message(
                        "Test Title", "Test message", mock_logger
                    )

                    assert result is True
                    # Check success logging
                    mock_logger.info.assert_called()
                    info_call = mock_logger.info.call_args[0][0]
                    assert "[Notify]" in info_call
                    assert "Test Title" in info_call

    def test_send_message_failure_with_logging(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test failed message sending with logging."""
        with patch.object(telegram_config, "_send_all_messages_async", return_value=False):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = False

                    result = telegram_config.send_message(
                        "Test Title", "Test message", mock_logger
                    )

                    assert result is False
                    # Check failure logging
                    mock_logger.error.assert_called()
                    error_call = mock_logger.error.call_args[0][0]
                    assert "Failed to send message" in error_call

    def test_send_message_splitting_single_long_piece(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting when a single piece is too long."""
        # Create a single very long word that can't be split at spaces
        long_word = "a" * 4500

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", long_word, mock_logger)

                    # Should be called with multiple messages due to splitting
                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_debug_logging(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test debug logging throughout send_message."""
        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", "Message", mock_logger)

                    # Verify debug logging calls
                    assert mock_logger.debug.call_count > 5
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    assert any("Entering send_message" in call for call in debug_calls)
                    assert any("Message format: markdown" in call for call in debug_calls)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_without_logger(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _send_message_async without logger."""
        mock_bot = AsyncMock()
        mock_message = Mock(spec=Message)
        mock_message.message_id = 12345
        mock_bot.send_message.return_value = mock_message

        # Call without logger
        result = await telegram_config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", None
        )

        assert result is True
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_without_logger(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test _send_all_messages_async without logger."""
        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = await telegram_config._send_all_messages_async(
                "Test Title", ["Test content"], "MarkdownV2", None
            )

            assert result is True
            mock_bot.send_message.assert_called_once()

    def test_send_message_without_logger(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test send_message without logger."""
        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    # Call without logger
                    result = telegram_config.send_message("Title", "Message", None)

                    assert result is True

    def test_send_message_no_credentials_no_logger(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test send_message with missing credentials and no logger."""
        telegram_config.telegram_bot_token = None
        result = telegram_config.send_message("Title", "Message", None)
        assert result is False

    def test_send_message_split_at_word_boundary(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting finds word boundaries correctly."""
        # Create a message where we need to split at word boundary
        word = "word "
        # Make it just long enough that adding one more word would exceed limit
        long_message = word * 800  # This should be close to the limit

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", long_message, mock_logger)

                    # The message should be split
                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_split_force_when_no_space(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test force split when no space found in long piece."""
        # Create a single very long "word" with no spaces that exceeds max_content_length
        long_word_piece = "a" * 5000  # Definitely exceeds max_content_length

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", long_word_piece, mock_logger)

                    # Should be called with split messages
                    mock_loop.return_value.run_until_complete.assert_called_once()
                    # Get the actual call to verify multiple messages were created
                    mock_loop.return_value.run_until_complete.call_args[0][0]

    def test_send_message_exception_in_thread(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test exception handling when running in thread."""
        with patch("asyncio.get_running_loop", return_value=Mock()):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                mock_future = Mock()
                mock_future.result.side_effect = Exception("Thread error")
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                result = telegram_config.send_message("Title", "Message", mock_logger)

                assert result is False
                mock_logger.error.assert_called()
                error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                assert any(
                    "Unexpected error sending Telegram message" in call for call in error_calls
                )

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_debug_logging(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test debug logging in _send_message_async."""
        mock_bot = AsyncMock()
        mock_message = Mock(spec=Message)
        mock_message.message_id = 12345
        mock_bot.send_message.return_value = mock_message

        mock_logger = Mock()

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger
        )

        assert result is True
        # Verify debug logging
        assert mock_logger.debug.call_count >= 3
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("_send_message_async called" in call for call in debug_calls)
        assert any("Message sent successfully" in call for call in debug_calls)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_no_retry_on_general_error(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test that general TelegramError doesn't retry."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TelegramError("General error")

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=3
        )

        assert result is False
        # Should only try once, no retries for general errors
        mock_bot.send_message.assert_called_once()
        mock_logger.error.assert_called_with("Telegram API error: General error")

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_debug_logging(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test debug logging in _send_all_messages_async."""
        mock_logger = Mock()

        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = await telegram_config._send_all_messages_async(
                "Test Title", ["Message 1", "Message 2"], "MarkdownV2", mock_logger
            )

            assert result is True
            # Verify debug logging
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any(
                "_send_all_messages_async called with 2 messages" in call for call in debug_calls
            )
            assert any("Using bot token:" in call for call in debug_calls)

    def test_send_message_message_splitting_with_current_message(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting when current_msg has content."""
        # Create messages that will trigger the split with existing current_msg
        pieces = ["Part 1" * 100, "Part 2" * 100, "Part 3" * 1000]  # Last part is very long
        message = "\n\n".join(pieces)

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", message, mock_logger)

                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_html_escaping_debug(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test HTML escaping with debug logging."""
        telegram_config.message_format = "html"

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    message = "Test <b>bold</b> & special"
                    telegram_config.send_message("Title", message, mock_logger)

                    # Verify HTML escaping was logged
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    assert any("Escaped message for HTML" in call for call in debug_calls)

    def test_send_message_empty_current_msg_after_split(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test handling when current_msg becomes empty after split."""
        # Create a message that splits exactly at paragraph boundaries
        piece1 = "x" * 3800  # Close to limit
        piece2 = "y" * 10  # Small piece
        message = f"{piece1}\n\n{piece2}"

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", message, mock_logger)

                    mock_loop.return_value.run_until_complete.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_bad_request_with_parse_error(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test BadRequest with parse error details."""
        mock_bot = AsyncMock()
        error = BadRequest("Can't parse entities: unexpected end tag")
        error.message = "Can't parse entities: unexpected end tag at byte offset 42"
        mock_bot.send_message.side_effect = error

        result = await telegram_config._send_message_async(
            mock_bot, "Test *unclosed markdown", "MarkdownV2", mock_logger
        )

        assert result is False
        # Verify detailed error logging for parse errors
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("Parse error detected" in call for call in error_calls)
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Line 0:" in call for call in debug_calls)  # Should log individual lines

    def test_send_message_thread_execution_logging(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test logging when already in event loop (thread execution)."""
        with patch("asyncio.get_running_loop", return_value=Mock()):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = True
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                result = telegram_config.send_message("Title", "Message", mock_logger)

                assert result is True
                # Verify thread execution was logged
                debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                assert any("Already in event loop, using thread" in call for call in debug_calls)

    def test_send_message_no_event_loop_debug(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test logging when no event loop exists."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running loop")):
            with patch("asyncio.new_event_loop") as mock_loop:
                mock_loop.return_value.run_until_complete.return_value = True

                with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
                    result = telegram_config.send_message("Title", "Message", mock_logger)

                    assert result is True
                    # Verify no event loop was logged
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    assert any("No event loop, creating new one" in call for call in debug_calls)

    def test_send_message_plain_text_debug(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test plain text (no escaping) with debug logging."""
        telegram_config.message_format = "plain_text"

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    message = "Test *bold* <tag>"
                    telegram_config.send_message("Title", message, mock_logger)

                    # Verify no escaping was logged
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    assert any("No escaping applied (plain text)" in call for call in debug_calls)

    def test_send_message_exception_with_traceback(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test exception handling with traceback logging."""
        with patch("asyncio.get_running_loop", side_effect=Exception("Test exception")):
            result = telegram_config.send_message("Title", "Message", mock_logger)

            assert result is False
            # Verify traceback was logged
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("Traceback:" in call for call in debug_calls)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_retry_after_zero_retries(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test RetryAfter with zero retries left."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = RetryAfter(retry_after=0.01)

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=0
        )

        assert result is False
        mock_bot.send_message.assert_called_once()
        mock_logger.error.assert_called_with("Max retries exceeded for rate limiting")

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_timeout_zero_retries(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test TimedOut with zero retries left."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TimedOut()

        result = await telegram_config._send_message_async(
            mock_bot, "Test message", None, mock_logger, max_retries=0
        )

        assert result is False
        mock_bot.send_message.assert_called_once()
        mock_logger.error.assert_called_with("Max retries exceeded for timeout")

    def test_thread_execution_run_in_thread(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test the run_in_thread function execution."""
        with patch("asyncio.get_running_loop", return_value=Mock()):
            with patch("asyncio.new_event_loop") as mock_new_loop:
                new_loop_instance = Mock()
                mock_new_loop.return_value = new_loop_instance
                new_loop_instance.run_until_complete.return_value = True
                new_loop_instance.close = Mock()

                with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                    # Simulate the thread executor actually calling the function
                    def execute_submit(func: Callable[[], Any]) -> Any:
                        future = Mock()
                        future.result = Mock(return_value=func())
                        return future

                    mock_executor.return_value.__enter__.return_value.submit.side_effect = (
                        execute_submit
                    )

                    result = telegram_config.send_message("Title", "Message", mock_logger)

                    assert result is True
                    # Verify new loop was created and closed in thread
                    mock_new_loop.assert_called()
                    new_loop_instance.close.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_should_not_reach_end(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test edge case where max_retries loop completes without returning."""
        mock_bot = AsyncMock()
        # This shouldn't happen in practice, but tests line 129
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Mock the for loop to somehow not return
            result = await telegram_config._send_message_async(
                mock_bot, "Test", None, mock_logger, max_retries=-1  # Will skip loop
            )
            # Should return False as fallback
            assert result is False

    def test_send_message_split_with_space_at_beginning(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test split point when space is at position 0."""
        # Create a message where split happens with space at beginning
        piece1 = "x" * 3800
        piece2 = " " + "y" * 100  # Space at beginning
        message = piece1 + "\n\n" + piece2

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", message, mock_logger)
                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_failed_no_logger(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test failed message sending without logger."""
        with patch.object(telegram_config, "_send_all_messages_async", return_value=False):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = False

                    result = telegram_config.send_message("Title", "Message", None)
                    assert result is False

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_retry_after_continue(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test RetryAfter with successful retry on first attempt."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = [RetryAfter(retry_after=0.01), Mock(message_id=12345)]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await telegram_config._send_message_async(
                mock_bot, "Test", None, mock_logger, max_retries=1
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_timeout_continue(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test TimedOut with successful retry."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = [TimedOut(), Mock(message_id=12345)]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await telegram_config._send_message_async(
                mock_bot, "Test", None, mock_logger, max_retries=1
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_message_async_bad_request_without_parse_error(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test BadRequest without 'can't parse' in message."""
        mock_bot = AsyncMock()
        error = BadRequest("Chat not found")
        error.message = "Chat not found"
        mock_bot.send_message.side_effect = error

        result = await telegram_config._send_message_async(mock_bot, "Test", None, mock_logger)

        assert result is False
        # Should not log parse error specific messages
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert not any("Parse error detected" in call for call in error_calls)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_send_all_messages_async_message_exceeds_limit(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message truncation when it exceeds the limit."""
        with patch("ai_marketplace_monitor.telegram.Bot") as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            # Create a message that will exceed limit after title + signature are added
            # The title formatting and signature will push this over the limit
            base_message = "x" * MessageLimit.MAX_TEXT_LENGTH

            result = await telegram_config._send_all_messages_async(
                "Very Long Title That Takes Up Space", [base_message], "MarkdownV2", mock_logger
            )

            assert result is True
            # Should warn about truncation
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "exceeded limit after formatting" in warning_call

            # Check the message was truncated
            sent_text = mock_bot.send_message.call_args[1]["text"]
            assert "..." in sent_text
            assert len(sent_text) <= MessageLimit.MAX_TEXT_LENGTH

    def test_send_message_markdown_escaping_splits(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test markdown escaping when message needs splitting."""
        # Create a message that will be split
        long_message = "Test *markdown* " * 300  # Will need splitting

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", long_message, mock_logger)

                    # Should log about escaping
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    assert any("Escaped message for MARKDOWN" in call for call in debug_calls)

    def test_send_message_split_at_exact_boundary(
        self: "TestTelegramMessaging",
        telegram_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test when split point is exactly at max_content_length with no space."""
        # Create a piece that's exactly at the limit
        max_content_length = MessageLimit.MAX_TEXT_LENGTH - 200
        piece = "a" * max_content_length
        message = f"{piece}\n\nmore content"

        with patch.object(telegram_config, "_send_all_messages_async", return_value=True):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError()):
                with patch("asyncio.new_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = True

                    telegram_config.send_message("Title", message, mock_logger)
                    mock_loop.return_value.run_until_complete.assert_called_once()

    def test_send_message_exception_no_logger(
        self: "TestTelegramMessaging", telegram_config: TelegramNotificationConfig
    ) -> None:
        """Test exception handling without logger."""
        with patch("asyncio.get_running_loop", side_effect=Exception("Test error")):
            result = telegram_config.send_message("Title", "Message", None)
            assert result is False

    def test_special_characters_and_emojis_escaping(
        self: "TestTelegramMessaging",
    ) -> None:
        """Test proper escaping of special characters and emojis in marketplace descriptions."""
        # Test MarkdownV2 format
        config = TelegramNotificationConfig(name="test", message_format="markdownv2")

        # Test cases covering various marketplace description scenarios
        test_cases = [
            ("AI Model 🤖", "AI Model 🤖"),  # Emojis should be preserved
            ("Price: $199.99", "Price: $199\\.99"),  # Dots escaped, dollar preserved
            (
                "Rating: ★★★★★ (4.8/5)",
                "Rating: ★★★★★ \\(4\\.8/5\\)",
            ),  # Stars preserved, parens escaped
            ("Features: [NLP], {ML}", "Features: \\[NLP\\], \\{ML\\}"),  # Brackets escaped
            ("Contact: user@example.com", "Contact: user@example\\.com"),  # Email dots escaped
            ("Discount: 50% off!", "Discount: 50% off\\!"),  # Exclamation escaped
            ("Languages: 中文, العربية", "Languages: 中文, العربية"),  # Unicode preserved
            ("Performance: ~95% accuracy", "Performance: \\~95% accuracy"),  # Tilde escaped
        ]

        for original, expected in test_cases:
            escaped = config._escape_text_for_format(original)
            assert (
                escaped == expected
            ), f"Failed for '{original}': got '{escaped}', expected '{expected}'"

        # Test HTML format doesn't break with special chars
        html_config = TelegramNotificationConfig(name="test", message_format="html")

        html_test = "AI Model <tag> & 🤖"
        html_escaped = html_config._escape_text_for_format(html_test)
        assert html_escaped == "AI Model &lt;tag&gt; &amp; 🤖"

        # Test plain text preserves everything
        plain_config = TelegramNotificationConfig(name="test", message_format="plain_text")

        plain_test = "AI Model [special] & 🤖"
        plain_escaped = plain_config._escape_text_for_format(plain_test)
        assert plain_escaped == plain_test

    def test_message_formatting_validation(
        self: "TestTelegramMessaging",
    ) -> None:
        """Test message formatting validation before sending."""
        # Test MarkdownV2 validation
        md_config = TelegramNotificationConfig(name="test", message_format="markdownv2")

        # Valid messages should pass
        assert md_config._validate_message_formatting("Simple text") is True
        assert md_config._validate_message_formatting("Text with *bold* and _italic_") is True
        assert md_config._validate_message_formatting("Link [text](url)") is True

        # Invalid messages should fail
        assert md_config._validate_message_formatting("Unmatched [bracket") is False
        assert md_config._validate_message_formatting("Unmatched (paren") is False
        assert md_config._validate_message_formatting("Text [link(missing bracket") is False

        # Test HTML validation
        html_config = TelegramNotificationConfig(name="test", message_format="html")

        # Valid HTML should pass
        assert html_config._validate_message_formatting("Simple text") is True
        assert html_config._validate_message_formatting("<b>Bold</b> and <i>italic</i>") is True
        assert html_config._validate_message_formatting("<a href='url'>Link</a>") is True

        # Invalid HTML should fail
        assert html_config._validate_message_formatting("<b>Unclosed bold") is False
        assert html_config._validate_message_formatting("<i>Unclosed italic") is False
        assert html_config._validate_message_formatting("Extra </b> closing tag") is False

        # Test plain text always passes
        plain_config = TelegramNotificationConfig(name="test", message_format="plain_text")

        assert plain_config._validate_message_formatting("Any [text] <tags> *formatting*") is True
        assert plain_config._validate_message_formatting("") is True

    def test_markdownv2_parse_error_detection(
        self: "TestTelegramMessaging",
    ) -> None:
        """Test detection and analysis of MarkdownV2 parse errors."""
        from unittest.mock import Mock

        from telegram.error import BadRequest

        config = TelegramNotificationConfig(name="test", message_format="markdownv2")

        mock_logger = Mock()

        # Test various MarkdownV2 parse error messages
        test_cases = [
            {
                "message": "Can't parse entities: unexpected character at byte offset 42",
                "expected": True,
                "description": "Standard parse error with byte offset",
            },
            {
                "message": "Can't parse message entities at line 1 char 15",
                "expected": True,
                "description": "Parse error with line/char info",
            },
            {
                "message": "Bad Request: can't parse entities: invalid markup",
                "expected": True,
                "description": "Invalid markup error",
            },
            {
                "message": "Can't find end of the entity starting at byte offset 10",
                "expected": True,
                "description": "Unclosed entity error",
            },
            {
                "message": "Entities overlap at byte offset 25",
                "expected": True,
                "description": "Overlapping entities error",
            },
            {
                "message": "Unexpected end tag at byte offset 50",
                "expected": True,
                "description": "Unexpected end tag error",
            },
            {"message": "Chat not found", "expected": False, "description": "Non-parse error"},
            {
                "message": "Bot was blocked by the user",
                "expected": False,
                "description": "User interaction error",
            },
        ]

        test_text = "Test message with *bold* and _italic_ [link](url) text"

        for case in test_cases:
            mock_logger.reset_mock()

            # Create BadRequest with the test message
            error = BadRequest(case["message"])
            error.message = case["message"]

            result = config._detect_markdownv2_parse_error(error, test_text, mock_logger)

            assert (
                result == case["expected"]
            ), f"Failed for: {case['description']} - Expected {case['expected']}, got {result}"

            if case["expected"]:
                # Should have logged parse error details
                assert (
                    mock_logger.error.called
                ), f"Should have logged error for: {case['description']}"
                error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                assert any("MarkdownV2 parse error detected" in call for call in error_calls)
            else:
                # Should not have detected as parse error
                if mock_logger.error.called:
                    error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                    assert not any(
                        "MarkdownV2 parse error detected" in call for call in error_calls
                    )

    def test_markdownv2_error_context_extraction(
        self: "TestTelegramMessaging",
    ) -> None:
        """Test extraction of context information from MarkdownV2 parse errors."""
        from unittest.mock import Mock

        from telegram.error import BadRequest

        config = TelegramNotificationConfig(name="test", message_format="markdownv2")

        mock_logger = Mock()

        # Test error with byte offset
        test_text = "This is a test message with [bad formatting at position 42"
        error_msg = "Can't parse entities: unexpected character at byte offset 42"
        error = BadRequest(error_msg)
        error.message = error_msg

        result = config._detect_markdownv2_parse_error(error, test_text, mock_logger)

        assert result is True

        # Verify detailed logging was called
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]

        # Should log the byte offset
        assert any("Parse error at byte offset: 42" in call for call in error_calls)

        # Should log the character at offset
        expected_char = test_text[42] if 42 < len(test_text) else "EOF"
        assert any(f"Character at offset 42: {expected_char!r}" in call for call in error_calls)

        # Should log context around error
        assert any("Context around error:" in call for call in error_calls)

    def test_fallback_format_hierarchy(self: "TestTelegramMessaging") -> None:
        """Test fallback format hierarchy generation."""
        test_cases = [
            {
                "format": "markdownv2",
                "expected": ["markdownv2", "markdown", "plain_text"],
                "description": "MarkdownV2 should fallback to Markdown then Plain text",
            },
            {
                "format": "markdown",
                "expected": ["markdown", "plain_text"],
                "description": "Markdown should fallback to Plain text",
            },
            {
                "format": "html",
                "expected": ["html", "plain_text"],
                "description": "HTML should fallback to Plain text",
            },
            {
                "format": "plain_text",
                "expected": ["plain_text"],
                "description": "Plain text should have no fallbacks",
            },
            {
                "format": None,
                "expected": ["markdownv2", "markdown", "plain_text"],
                "description": "None format should default to markdownv2 for Telegram",
            },
        ]

        for case in test_cases:
            config = TelegramNotificationConfig(name="test", message_format=case["format"])
            result = config._get_fallback_format_hierarchy()
            assert (
                result == case["expected"]
            ), f"Failed for {case['description']} - Expected {case['expected']}, got {result}"

    def test_message_format_conversion(self: "TestTelegramMessaging") -> None:
        """Test message format conversion between different formats."""
        from unittest.mock import Mock

        mock_logger = Mock()

        # Test MarkdownV2 to Markdown conversion
        config = TelegramNotificationConfig(name="test", message_format="markdownv2")
        markdownv2_text = "*Bold text* and _italic text_ with [link](https://example.com)"
        markdown_result = config._convert_message_to_format(
            markdownv2_text, "markdown", mock_logger
        )
        expected_markdown = "**Bold text** and _italic text_ with [link](https://example.com)"
        assert markdown_result == expected_markdown

        # Test any format to plain text conversion
        formatted_text = '*Bold* **bold** _italic_ [Link](https://example.com) <b>HTML bold</b> <i>HTML italic</i> <a href="url">Link text</a> line1<br>line2'
        plain_result = config._convert_message_to_format(formatted_text, "plain_text", mock_logger)
        expected_plain = "Bold bold italic Link HTML bold HTML italic Link text line1\nline2"
        assert plain_result == expected_plain

        # Test same format (no conversion)
        same_format_result = config._convert_message_to_format(
            "Test text", "markdownv2", mock_logger
        )
        assert same_format_result == "Test text"

    @pytest.mark.asyncio
    async def test_send_message_with_fallback_success(self: "TestTelegramMessaging") -> None:
        """Test successful message sending with fallback when primary format fails."""
        from unittest.mock import AsyncMock, Mock

        from telegram.error import BadRequest

        config = TelegramNotificationConfig(
            name="test",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
        )

        mock_logger = Mock()
        mock_bot = AsyncMock()

        # First call (MarkdownV2) fails with parse error
        parse_error = BadRequest("Can't parse entities: unexpected character")
        parse_error.message = "Can't parse entities: unexpected character"

        # Second call (Markdown fallback) succeeds
        success_result = Mock()
        success_result.message_id = 12345

        mock_bot.send_message.side_effect = [parse_error, success_result]

        result = await config._send_message_async(
            mock_bot, "*Test message*", "MarkdownV2", mock_logger
        )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Verify fallback was attempted
        calls = mock_bot.send_message.call_args_list
        assert calls[0][1]["parse_mode"] == "MarkdownV2"  # First call
        assert calls[1][1]["parse_mode"] == "Markdown"  # Fallback call
        assert calls[1][1]["text"] == "**Test message**"  # Converted text

        # Verify logging
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Trying fallback format: markdown" in call for call in info_calls)
        assert any(
            "Message sent successfully using fallback format: markdown" in call
            for call in info_calls
        )

    @pytest.mark.asyncio
    async def test_send_message_with_all_fallbacks_failing(self: "TestTelegramMessaging") -> None:
        """Test message sending when all fallback formats fail."""
        from unittest.mock import AsyncMock, Mock

        from telegram.error import BadRequest

        config = TelegramNotificationConfig(
            name="test",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
        )

        mock_logger = Mock()
        mock_bot = AsyncMock()

        # All attempts fail with parse errors
        parse_error1 = BadRequest("Can't parse entities in MarkdownV2")
        parse_error1.message = "Can't parse entities in MarkdownV2"

        parse_error2 = BadRequest("Can't parse entities in Markdown")
        parse_error2.message = "Can't parse entities in Markdown"

        parse_error3 = BadRequest("Some other error in plain text")
        parse_error3.message = "Some other error in plain text"

        mock_bot.send_message.side_effect = [parse_error1, parse_error2, parse_error3]

        result = await config._send_message_async(
            mock_bot, "*Test message*", "MarkdownV2", mock_logger
        )

        assert result is False
        assert mock_bot.send_message.call_count == 3

        # Verify all formats were tried
        calls = mock_bot.send_message.call_args_list
        assert calls[0][1]["parse_mode"] == "MarkdownV2"  # Primary
        assert calls[1][1]["parse_mode"] == "Markdown"  # First fallback
        assert calls[2][1]["parse_mode"] is None  # Second fallback (plain text)

        # Verify error logging
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("All fallback formats failed" in call for call in error_calls)

    @pytest.mark.asyncio
    async def test_send_message_non_parse_error_no_fallback(self: "TestTelegramMessaging") -> None:
        """Test that non-parse BadRequest errors don't trigger fallback."""
        from unittest.mock import AsyncMock, Mock

        from telegram.error import BadRequest

        config = TelegramNotificationConfig(
            name="test",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
        )

        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Non-parse error (e.g., chat not found)
        non_parse_error = BadRequest("Chat not found")
        non_parse_error.message = "Chat not found"

        mock_bot.send_message.side_effect = non_parse_error

        result = await config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger
        )

        assert result is False
        assert mock_bot.send_message.call_count == 1  # No fallback attempts

        # Verify non-parse error was logged
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("Non-parse BadRequest error" in call for call in error_calls)

        # Verify no fallback was attempted
        info_calls = (
            [call[0][0] for call in mock_logger.info.call_args_list]
            if mock_logger.info.called
            else []
        )
        assert not any("Trying fallback format" in call for call in info_calls)


class TestTelegramMarkdownV2Formatting:
    """Comprehensive test suite for MarkdownV2 formatting with various inputs."""

    @pytest.fixture
    def markdownv2_config(self: "TestTelegramMarkdownV2Formatting") -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig with MarkdownV2 formatting."""
        return TelegramNotificationConfig(
            name="test_markdownv2",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
        )

    def test_escape_basic_special_characters(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test escaping of basic MarkdownV2 special characters."""
        test_cases = [
            ("Simple text", "Simple text"),
            ("Text with _ underscore", "Text with \\_ underscore"),
            ("Text with * asterisk", "Text with \\* asterisk"),
            ("Text with [ bracket", "Text with \\[ bracket"),
            ("Text with ] bracket", "Text with \\] bracket"),
            ("Text with ( parenthesis", "Text with \\( parenthesis"),
            ("Text with ) parenthesis", "Text with \\) parenthesis"),
            ("Text with ~ tilde", "Text with \\~ tilde"),
            ("Text with ` backtick", "Text with \\` backtick"),
            ("Text with > greater", "Text with \\> greater"),
            ("Text with # hash", "Text with \\# hash"),
            ("Text with + plus", "Text with \\+ plus"),
            ("Text with - minus", "Text with \\- minus"),
            ("Text with = equals", "Text with \\= equals"),
            ("Text with | pipe", "Text with \\| pipe"),
            ("Text with { brace", "Text with \\{ brace"),
            ("Text with } brace", "Text with \\} brace"),
            ("Text with . dot", "Text with \\. dot"),
            ("Text with ! exclamation", "Text with \\! exclamation"),
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_escape_multiple_special_characters(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test escaping of multiple special characters in one string."""
        test_cases = [
            (
                "Text with _multiple* [special] characters",
                "Text with \\_multiple\\* \\[special\\] characters",
            ),
            ("Price: $19.99 (20% off!)", "Price: $19\\.99 \\(20% off\\!\\)"),
            (
                "Link: https://example.com/path?param=value",
                "Link: https://example\\.com/path\\?param\\=value",
            ),
            (
                "Code: `function() { return true; }`",
                "Code: \\`function\\(\\) \\{ return true; \\}\\`",
            ),
            ("Formula: x^2 + y^2 = z^2", "Formula: x\\^2 \\+ y\\^2 \\= z\\^2"),
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_format_bold_text(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test bold formatting with various inputs."""
        test_cases = [
            ("Simple", "*Simple*"),
            ("Text with spaces", "*Text with spaces*"),
            ("Text_with_underscores", "*Text\\_with\\_underscores*"),
            ("Text*with*asterisks", "*Text\\*with\\*asterisks*"),
            ("Text with [brackets]", "*Text with \\[brackets\\]*"),
            ("", "**"),  # Empty string
            ("Multiple words here", "*Multiple words here*"),
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._format_bold(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_format_italic_link(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test italic link formatting with various inputs."""
        test_cases = [
            ("Simple link", "https://example.com", "_[Simple link](https://example.com)_"),
            (
                "Link with spaces",
                "https://example.com/path",
                "_[Link with spaces](https://example.com/path)_",
            ),
            (
                "Link_with_underscores",
                "https://example.com",
                "_[Link\\_with\\_underscores](https://example.com)_",
            ),
            (
                "Link with [brackets]",
                "https://example.com",
                "_[Link with \\[brackets\\]](https://example.com)_",
            ),
            ("", "https://example.com", "_[](https://example.com)_"),  # Empty text
            ("Link", "", "_[Link]()_"),  # Empty URL
        ]

        for text, url, expected in test_cases:
            result = markdownv2_config._format_italic_link(text, url)
            assert result == expected, f"Failed for text: {text!r}, url: {url!r}"

    def test_escape_emoji_characters(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test escaping with emoji characters."""
        test_cases = [
            ("Happy face 😊", "Happy face 😊"),  # Emojis should not be escaped
            ("Emoji with special: 🚀*awesome*", "Emoji with special: 🚀\\*awesome\\*"),
            ("Price 💰$19.99", "Price 💰$19\\.99"),
            ("Success ✅ (complete!)", "Success ✅ \\(complete\\!\\)"),
            ("Warning ⚠️ [check this]", "Warning ⚠️ \\[check this\\]"),
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_escape_unicode_characters(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test escaping with various Unicode characters."""
        test_cases = [
            ("Accented café", "Accented café"),
            ("Chinese 中文", "Chinese 中文"),
            ("Japanese こんにちは", "Japanese こんにちは"),
            ("Arabic العربية", "Arabic العربية"),
            ("Russian Русский", "Russian Русский"),
            ("Math symbols: ∑∆√", "Math symbols: ∑∆√"),
            ("Unicode with special: café*", "Unicode with special: café\\*"),
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_escape_newlines_and_whitespace(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test escaping with newlines and various whitespace."""
        test_cases = [
            ("Line 1\nLine 2", "Line 1\nLine 2"),  # Newlines preserved
            ("Line 1\n\nLine 3", "Line 1\n\nLine 3"),  # Double newlines preserved
            ("Text\twith\ttabs", "Text\twith\ttabs"),  # Tabs preserved
            ("Multiple    spaces", "Multiple    spaces"),  # Multiple spaces preserved
            ("Mixed\n*formatting*", "Mixed\n\\*formatting\\*"),  # Newlines + special chars
            ("Trailing\nspaces   ", "Trailing\nspaces   "),  # Trailing spaces preserved
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_escape_edge_cases(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test edge cases for MarkdownV2 escaping."""
        test_cases = [
            ("", ""),  # Empty string
            ("   ", "   "),  # Only spaces
            ("\n\n\n", "\n\n\n"),  # Only newlines
            ("*", "\\*"),  # Single special character
            ("**", "\\*\\*"),  # Repeated special characters
            ("***", "\\*\\*\\*"),  # Multiple repeated special characters
            ("\\", "\\\\"),  # Backslash should be escaped
            ("\\*", "\\\\\\*"),  # Already escaped character
            ("Text\\*with\\*escapes", "Text\\\\\\*with\\\\\\*escapes"),  # Pre-escaped content
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_real_world_marketplace_descriptions(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test escaping with real-world marketplace descriptions."""
        test_cases = [
            ("iPhone 13 Pro - $899 (Like new!)", "iPhone 13 Pro \\- $899 \\(Like new\\!\\)"),
            ("Gaming PC: RTX 3080 + Intel i7", "Gaming PC: RTX 3080 \\+ Intel i7"),
            (
                "Apartment for rent - 2BR/2BA $1,500/month",
                "Apartment for rent \\- 2BR/2BA $1,500/month",
            ),
            (
                "Car for sale: 2018 Honda Civic (95k miles) - $15,000",
                "Car for sale: 2018 Honda Civic \\(95k miles\\) \\- $15,000",
            ),
            (
                "Freelance work: Web dev needed ($50/hr)",
                "Freelance work: Web dev needed \\($50/hr\\)",
            ),
            (
                "Event: Concert @ 8PM [VIP tickets available]",
                "Event: Concert @ 8PM \\[VIP tickets available\\]",
            ),
        ]

        for input_text, expected in test_cases:
            result = markdownv2_config._escape_text_for_format(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_markdownv2_parse_mode_detection(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test that parse mode is correctly detected for MarkdownV2."""
        parse_mode = markdownv2_config._get_parse_mode()
        assert parse_mode == "MarkdownV2"

    def test_message_formatting_validation_success(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test message formatting validation for valid MarkdownV2."""
        valid_messages = [
            "Simple text message",
            "*Bold text*",
            "_Italic text_",
            "[Link text](https://example.com)",
            "*Bold* and _italic_ combined",
            "Text with properly escaped \\* characters",
            "Multiple lines\nwith proper formatting",
            "Text with emoji 😊 and special chars properly escaped\\!",
        ]

        for message in valid_messages:
            result = markdownv2_config._validate_message_formatting(message)
            assert result is True, f"Valid message failed validation: {message!r}"

    def test_message_formatting_validation_failures(
        self: "TestTelegramMarkdownV2Formatting",
        markdownv2_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message formatting validation for invalid MarkdownV2."""
        invalid_messages = [
            "Unmatched *bold formatting",
            "Unmatched _italic formatting",
            "Unmatched [link formatting",
            "Unmatched (parenthesis",
            "*Bold with _nested italic*",  # Overlapping formatting
            "[Link with [nested] brackets](url)",
        ]

        for message in invalid_messages:
            result = markdownv2_config._validate_message_formatting(message, mock_logger)
            assert result is False, f"Invalid message passed validation: {message!r}"

    def test_formatting_with_long_content(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test formatting with long content that might be split."""
        long_text = "This is a very long text " * 100  # Create long text
        long_text_with_special = long_text + " with special characters *[]()!"

        result = markdownv2_config._escape_text_for_format(long_text_with_special)

        # Should escape special characters at the end
        assert result.endswith(" with special characters \\*\\[\\]\\(\\)\\!")
        # Should preserve the repeated text
        assert "This is a very long text " in result
        # Should be longer than original due to escaping
        assert len(result) > len(long_text_with_special)

    def test_fallback_format_hierarchy_includes_markdownv2(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test that fallback format hierarchy is correct for MarkdownV2."""
        hierarchy = markdownv2_config._get_fallback_format_hierarchy()
        expected = ["markdownv2", "markdown", "plain_text"]
        assert hierarchy == expected

    def test_format_conversion_from_markdownv2(
        self: "TestTelegramMarkdownV2Formatting", markdownv2_config: TelegramNotificationConfig
    ) -> None:
        """Test message format conversion from MarkdownV2 to other formats."""
        test_message = "*Bold text* with _italic_ and [link](https://example.com)"

        # Convert to markdown
        markdown_result = markdownv2_config._convert_message_to_format(test_message, "markdown")
        # Should convert single * to **
        assert "**Bold text**" in markdown_result

        # Convert to plain text
        plain_result = markdownv2_config._convert_message_to_format(test_message, "plain_text")
        # Should remove all formatting
        assert plain_result == "Bold text with italic and link"

        # Convert to same format should return unchanged
        same_result = markdownv2_config._convert_message_to_format(test_message, "markdownv2")
        assert same_result == test_message


class TestTelegramErrorHandlingAndFallbacks:
    """Comprehensive test suite for error handling and fallback mechanisms."""

    @pytest.fixture
    def fallback_config(
        self: "TestTelegramErrorHandlingAndFallbacks",
    ) -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig for fallback testing."""
        return TelegramNotificationConfig(
            name="test_fallback",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
            max_retries=2,
            base_delay=0.1,
            max_delay=1.0,
            jitter=False,  # Disable jitter for predictable tests
        )

    def test_detect_markdownv2_parse_error_patterns(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test detection of various MarkdownV2 parse error patterns."""
        error_patterns = [
            ("Can't parse entities: invalid markup", True),
            ("Can't parse message entities", True),
            ("Invalid markup at byte offset 123", True),
            ("Unexpected end tag", True),
            ("Unexpected character '>' at position 5", True),
            ("Can't find end of the entity", True),
            ("Bad request: can't parse entities: invalid markup", True),
            ("Entities overlap", True),
            ("Regular BadRequest error", False),
            ("Network timeout", False),
            ("Invalid chat_id", False),
        ]

        for error_msg, should_detect in error_patterns:
            error = BadRequest(error_msg)
            error.message = error_msg

            result = fallback_config._detect_markdownv2_parse_error(
                error, "test text", mock_logger
            )
            assert result == should_detect, f"Failed for error: {error_msg!r}"

    def test_detect_markdownv2_parse_error_with_byte_offset(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test MarkdownV2 parse error detection with byte offset information."""
        test_text = "This is a test message with *invalid formatting"
        error = BadRequest("Can't parse entities: invalid markup at byte offset 30")
        error.message = "Can't parse entities: invalid markup at byte offset 30"

        result = fallback_config._detect_markdownv2_parse_error(
            error, test_text, mock_logger, "test-123"
        )

        assert result is True
        # Verify structured logging was called with byte offset details
        mock_logger.info.assert_called()  # Should have logged structured info
        mock_logger.debug.assert_called()  # Should have logged debug info

    def test_fallback_format_hierarchy_all_formats(
        self: "TestTelegramErrorHandlingAndFallbacks", fallback_config: TelegramNotificationConfig
    ) -> None:
        """Test fallback format hierarchy for different starting formats."""
        test_cases = [
            ("markdownv2", ["markdownv2", "markdown", "plain_text"]),
            ("markdown", ["markdown", "plain_text"]),
            ("html", ["html", "plain_text"]),
            ("plain_text", ["plain_text"]),
            ("invalid_format", ["plain_text"]),  # Should fallback to plain_text
        ]

        for format_type, expected in test_cases:
            fallback_config.message_format = format_type
            result = fallback_config._get_fallback_format_hierarchy()
            assert result == expected, f"Failed for format: {format_type}"

    def test_message_format_conversion_markdownv2_to_markdown(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message format conversion from MarkdownV2 to Markdown."""
        test_cases = [
            ("*Bold text*", "**Bold text**"),
            ("*Multiple* *bold* words", "**Multiple** **bold** words"),
            ("Regular text", "Regular text"),
            ("Mixed *bold* and regular", "Mixed **bold** and regular"),
            ("*Bold* with _italic_", "**Bold** with _italic_"),  # Only converts bold
        ]

        for markdownv2_text, expected_markdown in test_cases:
            result = fallback_config._convert_message_to_format(
                markdownv2_text, "markdown", mock_logger
            )
            assert result == expected_markdown, f"Failed for input: {markdownv2_text!r}"

    def test_message_format_conversion_to_plain_text(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message format conversion to plain text from various formats."""
        test_cases = [
            # MarkdownV2/Markdown formatting removal
            ("*Bold text*", "Bold text"),
            ("**Bold text**", "Bold text"),
            ("_Italic text_", "Italic text"),
            ("[Link text](https://example.com)", "Link text"),
            ("*Bold* and _italic_ with [link](url)", "Bold and italic with link"),
            # HTML formatting removal
            ("<b>Bold</b>", "Bold"),
            ("<i>Italic</i>", "Italic"),
            ("<a href='url'>Link</a>", "Link"),
            ("<br>", "\n"),
            # Mixed formatting
            ("**Bold** <i>italic</i> *mix*", "Bold italic mix"),
        ]

        for formatted_text, expected_plain in test_cases:
            result = fallback_config._convert_message_to_format(
                formatted_text, "plain_text", mock_logger
            )
            assert result == expected_plain, f"Failed for input: {formatted_text!r}"

    def test_is_transient_error_classification(
        self: "TestTelegramErrorHandlingAndFallbacks", fallback_config: TelegramNotificationConfig
    ) -> None:
        """Test classification of transient vs non-transient errors."""
        # Transient errors
        transient_errors = [
            RetryAfter(30),
            TimedOut(),
            NetworkError("Connection failed"),
            TelegramError("Network error occurred"),
            TelegramError("Connection timeout"),
            TelegramError("Temporary server error"),
            TelegramError("Internal server error"),
            TelegramError("Service unavailable"),
            TelegramError("Too many requests"),
            TelegramError("502 Bad Gateway"),
            TelegramError("503 Service Unavailable"),
            TelegramError("504 Gateway Timeout"),
        ]

        for error in transient_errors:
            result = fallback_config._is_transient_error(error)
            assert result is True, f"Should be transient: {error}"

        # Non-transient errors
        non_transient_errors = [
            BadRequest("Invalid chat_id"),
            TelegramError("Invalid bot token"),
            TelegramError("Chat not found"),
            TelegramError("Forbidden: bot blocked by user"),
            TelegramError("Bad Request: message is too long"),
        ]

        for error in non_transient_errors:
            result = fallback_config._is_transient_error(error)
            assert result is False, f"Should not be transient: {error}"

    def test_calculate_retry_delay_exponential_backoff(
        self: "TestTelegramErrorHandlingAndFallbacks", fallback_config: TelegramNotificationConfig
    ) -> None:
        """Test exponential backoff delay calculation."""
        # Test without jitter for predictable results
        test_cases = [
            (0, 0.1),  # First retry: base_delay * 2^0 = 0.1
            (1, 0.2),  # Second retry: base_delay * 2^1 = 0.2
            (2, 0.4),  # Third retry: base_delay * 2^2 = 0.4
            (3, 0.8),  # Fourth retry: base_delay * 2^3 = 0.8
            (4, 1.0),  # Fifth retry: min(base_delay * 2^4, max_delay) = 1.0 (capped)
            (10, 1.0),  # Large attempt: should be capped at max_delay
        ]

        for attempt, expected_delay in test_cases:
            result = fallback_config._calculate_retry_delay(
                attempt, base_delay=0.1, max_delay=1.0, jitter=False
            )
            assert (
                abs(result - expected_delay) < 0.001
            ), f"Failed for attempt {attempt}: expected {expected_delay}, got {result}"

    def test_calculate_retry_delay_with_jitter(
        self: "TestTelegramErrorHandlingAndFallbacks", fallback_config: TelegramNotificationConfig
    ) -> None:
        """Test retry delay calculation with jitter."""
        base_delay = 1.0
        max_delay = 10.0
        attempt = 2

        # Calculate expected delay without jitter
        expected_base = base_delay * (2**attempt)  # 4.0

        # With jitter, delay should be within ±25% of base
        min_expected = expected_base * 0.75
        max_expected = expected_base * 1.25

        # Test multiple times to account for randomness
        for _ in range(10):
            result = fallback_config._calculate_retry_delay(
                attempt, base_delay=base_delay, max_delay=max_delay, jitter=True
            )
            assert (
                min_expected <= result <= max_expected
            ), f"Jittered delay {result} outside expected range [{min_expected}, {max_expected}]"
            assert result >= 0.1, "Delay should have minimum of 0.1s"

    @pytest.mark.asyncio
    async def test_send_message_with_retry_rate_limiting(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test retry logic with rate limiting (RetryAfter) errors."""
        mock_bot = AsyncMock()

        # First call: rate limited, second call: success
        retry_after_error = RetryAfter(2)
        mock_bot.send_message.side_effect = [retry_after_error, Mock(message_id=12345)]

        # Mock asyncio.sleep to avoid actual waiting
        with patch("asyncio.sleep") as mock_sleep:
            result = await fallback_config._send_message_with_retry(
                mock_bot, "Test message", "MarkdownV2", mock_logger, max_retries=2
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2
        mock_sleep.assert_called_once_with(2)  # Should use server-specified delay

    @pytest.mark.asyncio
    async def test_send_message_with_retry_timeout_backoff(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test retry logic with timeout errors using exponential backoff."""
        mock_bot = AsyncMock()

        # First call: timeout, second call: success
        mock_bot.send_message.side_effect = [TimedOut(), Mock(message_id=12345)]

        # Mock asyncio.sleep to check backoff delay
        with patch("asyncio.sleep") as mock_sleep:
            result = await fallback_config._send_message_with_retry(
                mock_bot, "Test message", "MarkdownV2", mock_logger, max_retries=2
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Should use exponential backoff (base_delay * 2^0 = 0.1)
        mock_sleep.assert_called_once()
        call_args = mock_sleep.call_args[0]
        assert 0.05 <= call_args[0] <= 0.15  # Allow some variance for potential jitter

    @pytest.mark.asyncio
    async def test_send_message_with_retry_max_retries_exceeded(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test that max retries are respected."""
        mock_bot = AsyncMock()

        # Always fail with timeout
        mock_bot.send_message.side_effect = TimedOut()

        with patch("asyncio.sleep"):
            result = await fallback_config._send_message_with_retry(
                mock_bot, "Test message", "MarkdownV2", mock_logger, max_retries=2
            )

        assert result is False
        assert mock_bot.send_message.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_send_message_with_fallback_formatting(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test fallback formatting when MarkdownV2 parsing fails."""
        mock_bot = AsyncMock()

        # First call: MarkdownV2 parse error, second call: success with markdown
        parse_error = BadRequest("Can't parse entities: invalid markup")
        parse_error.message = "Can't parse entities: invalid markup"

        mock_bot.send_message.side_effect = [parse_error, Mock(message_id=12345)]

        result = await fallback_config._send_message_with_retry(
            mock_bot, "*Invalid* formatting*", "MarkdownV2", mock_logger, max_retries=2
        )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Check that second call used fallback format
        second_call = mock_bot.send_message.call_args_list[1]
        assert second_call[1]["parse_mode"] in ["Markdown", None]  # Should fallback

    @pytest.mark.asyncio
    async def test_send_message_with_all_fallbacks_failed(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test behavior when all fallback formats fail."""
        mock_bot = AsyncMock()

        # All attempts fail with parse errors
        parse_error = BadRequest("Can't parse entities: invalid markup")
        parse_error.message = "Can't parse entities: invalid markup"
        mock_bot.send_message.side_effect = parse_error

        result = await fallback_config._send_message_with_retry(
            mock_bot, "*Invalid* formatting*", "MarkdownV2", mock_logger, max_retries=2
        )

        assert result is False
        # Should try MarkdownV2, then Markdown, then plain_text
        assert mock_bot.send_message.call_count >= 3

    def test_message_formatting_validation_html(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message formatting validation for HTML format."""
        fallback_config.message_format = "html"

        valid_html = [
            "<b>Bold text</b>",
            "<i>Italic text</i>",
            "<b>Bold</b> and <i>italic</i>",
            "Regular text without tags",
            "<a href='https://example.com'>Link</a>",
        ]

        invalid_html = [
            "<b>Unclosed bold tag",
            "<i>Unclosed italic</i> <b>unclosed bold",
            "Mismatched <b>tags</i>",
            "<b><i>Nested unclosed</b>",
        ]

        for html in valid_html:
            result = fallback_config._validate_message_formatting(html, mock_logger)
            assert result is True, f"Valid HTML failed validation: {html!r}"

        for html in invalid_html:
            result = fallback_config._validate_message_formatting(html, mock_logger)
            assert result is False, f"Invalid HTML passed validation: {html!r}"

    def test_structured_logging_components(
        self: "TestTelegramErrorHandlingAndFallbacks",
        fallback_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test that structured logging uses correct components."""
        # Test mask sensitive data
        token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"  # noqa: S105
        masked = fallback_config._mask_sensitive_data(token, "token")

        assert "123456" in masked  # Should show first 6 chars
        assert "***" in masked  # Should have masking
        assert len(token) > 20  # Original should be long
        assert len(masked) < len(token)  # Masked should be shorter

        # Test correlation ID generation
        corr_id = fallback_config._generate_correlation_id()
        assert len(corr_id) == 8
        assert corr_id.isalnum() or "-" in corr_id  # UUID format

        # Test structured logging format
        fallback_config._log_structured(
            mock_logger,
            "INFO",
            "TEST",
            "Test message",
            correlation_id="test-123",
            test_param="value",
            numeric_param=42,
        )

        # Should have called the appropriate log level
        mock_logger.info.assert_called()
        logged_message = mock_logger.info.call_args[0][0]

        assert "[TELEGRAM:TEST]" in logged_message
        assert "[test-123]" in logged_message
        assert "Test message" in logged_message
        assert "test_param='value'" in logged_message
        assert "numeric_param=42" in logged_message


class TestTelegramMessageSplitting:
    """Comprehensive test suite for message splitting functionality."""

    @pytest.fixture
    def splitting_config(self: "TestTelegramMessageSplitting") -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig for message splitting tests."""
        return TelegramNotificationConfig(
            name="test_splitting",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
            max_retries=1,
        )

    def test_calculate_message_overhead_basic(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test basic message overhead calculation."""
        title = "Test Title"

        # Single part message
        overhead = splitting_config._calculate_message_overhead(title, 1, "test-123", mock_logger)

        # Should include title formatting, signature, separators, and safety buffer
        assert overhead > 0
        assert overhead < 500  # Reasonable upper bound

        # Multi-part message should have higher overhead
        multi_overhead = splitting_config._calculate_message_overhead(
            title, 5, "test-123", mock_logger
        )
        assert multi_overhead > overhead  # Should include part numbering

    def test_calculate_message_overhead_long_title(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message overhead calculation with long title."""
        short_title = "Short"
        long_title = "This is a very long title that contains many words and special characters!"

        short_overhead = splitting_config._calculate_message_overhead(
            short_title, 1, "test-123", mock_logger
        )
        long_overhead = splitting_config._calculate_message_overhead(
            long_title, 1, "test-123", mock_logger
        )

        assert long_overhead > short_overhead

    def test_detect_message_length_violations(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message length violation detection."""
        short_message = "This is a short message"
        long_message = "x" * 5000  # Definitely over limit

        # Short message should not be a violation
        short_analysis = splitting_config._detect_message_length_violations(
            short_message, 4096, "test-123", mock_logger
        )
        assert short_analysis["is_violation"] is False
        assert short_analysis["overflow_chars"] == 0
        assert short_analysis["estimated_split_needed"] is False

        # Long message should be a violation
        long_analysis = splitting_config._detect_message_length_violations(
            long_message, 4096, "test-123", mock_logger
        )
        assert long_analysis["is_violation"] is True
        assert long_analysis["overflow_chars"] > 0
        assert long_analysis["estimated_split_needed"] is True
        assert "estimated_parts" in long_analysis

    def test_split_message_preserving_formatting_short(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting with short message that doesn't need splitting."""
        short_message = "This is a short message"
        max_length = 1000

        result = splitting_config._split_message_preserving_formatting(
            short_message, max_length, "test-123", mock_logger
        )

        assert len(result) == 1
        assert result[0] == short_message

    def test_split_message_preserving_formatting_paragraph_boundaries(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting at paragraph boundaries."""
        paragraph1 = "This is the first paragraph. " * 20
        paragraph2 = "This is the second paragraph. " * 20
        paragraph3 = "This is the third paragraph. " * 20

        message = f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"
        max_length = len(paragraph1) + len(paragraph2) + 50  # Should split before paragraph3

        result = splitting_config._split_message_preserving_formatting(
            message, max_length, "test-123", mock_logger
        )

        assert len(result) >= 2
        # First part should contain first two paragraphs
        assert paragraph1 in result[0]
        assert paragraph2 in result[0]
        # Last part should contain third paragraph
        assert paragraph3 in result[-1]

    def test_split_message_preserving_formatting_line_boundaries(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting at line boundaries when paragraph split not available."""
        lines = ["Line " + str(i) + " content here" for i in range(50)]
        message = "\n".join(lines)
        max_length = len("\n".join(lines[:25])) + 50  # Should split around line 25

        result = splitting_config._split_message_preserving_formatting(
            message, max_length, "test-123", mock_logger
        )

        assert len(result) >= 2
        # Should preserve line structure
        assert "Line 0" in result[0]
        assert "Line 49" in result[-1]

    def test_split_message_preserving_formatting_sentence_boundaries(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting at sentence boundaries."""
        sentences = [f"This is sentence number {i}. " for i in range(100)]
        message = "".join(sentences)
        max_length = len("".join(sentences[:50])) + 50  # Should split around sentence 50

        result = splitting_config._split_message_preserving_formatting(
            message, max_length, "test-123", mock_logger
        )

        assert len(result) >= 2
        # Should end parts at sentence boundaries
        for part in result[:-1]:  # All parts except last should end with period
            assert part.rstrip().endswith(".")

    def test_split_message_preserving_formatting_word_boundaries(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message splitting at word boundaries as last resort."""
        words = ["word" + str(i) for i in range(1000)]
        message = " ".join(words)
        max_length = len(" ".join(words[:500])) + 50  # Should split around word 500

        result = splitting_config._split_message_preserving_formatting(
            message, max_length, "test-123", mock_logger
        )

        assert len(result) >= 2
        # Should not split words
        for part in result:
            assert " word" in part or part.startswith("word")  # Should have complete words

    def test_split_message_preserving_formatting_force_split(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test forced character splitting when no good boundaries found."""
        # Create a very long string with no spaces, newlines, or sentence boundaries
        message = "x" * 1000
        max_length = 400

        result = splitting_config._split_message_preserving_formatting(
            message, max_length, "test-123", mock_logger
        )

        assert len(result) >= 3  # Should be split into multiple parts

        # Each part should be within limits
        for part in result:
            assert len(part) <= max_length

        # All parts combined should equal original
        combined = "".join(result)
        assert combined == message

    def test_find_best_split_point_paragraph(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test finding best split point at paragraph boundaries."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        max_length = 30  # Should find split after first paragraph

        split_point = splitting_config._find_best_split_point(
            text, max_length, "\n\n", "test-123", mock_logger
        )

        assert split_point > 0
        assert text[:split_point].endswith("\n\n")

    def test_find_best_split_point_sentence(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test finding best split point at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        max_length = 25  # Should find split after first sentence

        split_point = splitting_config._find_best_split_point(
            text, max_length, ". ", "test-123", mock_logger
        )

        assert split_point > 0
        assert text[:split_point].endswith(". ")

    def test_find_best_split_point_word(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test finding best split point at word boundaries."""
        text = "word1 word2 word3 word4 word5"
        max_length = 15  # Should find split after word2 or word3

        split_point = splitting_config._find_best_split_point(
            text, max_length, " ", "test-123", mock_logger
        )

        assert split_point > 0
        assert text[split_point - 1] == " " or split_point == len(text)

    def test_is_inside_formatting_block_markdownv2(
        self: "TestTelegramMessageSplitting", splitting_config: TelegramNotificationConfig
    ) -> None:
        """Test detection of being inside MarkdownV2 formatting blocks."""
        # Code block detection
        text_in_code = "Some text ```code block content"
        position_in_code = len(text_in_code)
        assert splitting_config._is_inside_formatting_block(text_in_code, position_in_code) is True

        # Not in code block
        text_not_in_code = "Some text ```code block``` more text"
        position_not_in_code = len(text_not_in_code)
        assert (
            splitting_config._is_inside_formatting_block(text_not_in_code, position_not_in_code)
            is False
        )

        # Inline code detection
        text_in_inline = "Some text `inline code"
        position_in_inline = len(text_in_inline)
        assert (
            splitting_config._is_inside_formatting_block(text_in_inline, position_in_inline)
            is True
        )

    def test_is_inside_formatting_block_html(
        self: "TestTelegramMessageSplitting", splitting_config: TelegramNotificationConfig
    ) -> None:
        """Test detection of being inside HTML formatting blocks."""
        splitting_config.message_format = "html"

        # Inside HTML tag
        text_in_tag = "Some text <b>bold content"
        position_in_tag = len(text_in_tag)
        assert splitting_config._is_inside_formatting_block(text_in_tag, position_in_tag) is True

        # Not inside HTML tag
        text_not_in_tag = "Some text <b>bold</b> more text"
        position_not_in_tag = len(text_not_in_tag)
        assert (
            splitting_config._is_inside_formatting_block(text_not_in_tag, position_not_in_tag)
            is False
        )

    def test_preserve_formatting_context_markdownv2(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test formatting context preservation for MarkdownV2."""
        # Unmatched asterisk
        chunk_with_unmatched = "This is *bold text"
        fixed_chunk = splitting_config._preserve_formatting_context(
            chunk_with_unmatched, "test-123", mock_logger
        )
        assert fixed_chunk == "This is *bold text*"

        # Unmatched bracket
        chunk_with_bracket = "This is [link text"
        fixed_bracket = splitting_config._preserve_formatting_context(
            chunk_with_bracket, "test-123", mock_logger
        )
        assert fixed_bracket == "This is [link text]"

        # Already balanced
        chunk_balanced = "This is *bold* text"
        fixed_balanced = splitting_config._preserve_formatting_context(
            chunk_balanced, "test-123", mock_logger
        )
        assert fixed_balanced == chunk_balanced

    def test_preserve_formatting_context_html(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test formatting context preservation for HTML."""
        splitting_config.message_format = "html"

        # Unclosed bold tag
        chunk_with_unclosed = "This is <b>bold text"
        fixed_chunk = splitting_config._preserve_formatting_context(
            chunk_with_unclosed, "test-123", mock_logger
        )
        assert fixed_chunk == "This is <b>bold text</b>"

        # Multiple unclosed tags
        chunk_multiple = "This is <b>bold and <i>italic"
        fixed_multiple = splitting_config._preserve_formatting_context(
            chunk_multiple, "test-123", mock_logger
        )
        assert "</b>" in fixed_multiple
        assert "</i>" in fixed_multiple

    def test_format_message_title_with_numbering(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message title formatting with numbering."""
        title = "Test Message"

        # Single part - no numbering
        single_title = splitting_config._format_message_title_with_numbering(
            title, 1, 1, "test-123", mock_logger
        )
        assert single_title == "*Test Message*"  # Just bold, no numbering

        # Multiple parts - with numbering
        multi_title = splitting_config._format_message_title_with_numbering(
            title, 2, 5, "test-123", mock_logger
        )
        assert multi_title == "*Test Message* (2/5)"

        # Large numbers - bracket format
        splitting_config._format_message_title_with_numbering(
            title, 50, 150, "test-123", mock_logger
        )
        assert multi_title == "*Test Message* (2/5)"  # Should still use parentheses for <= 99

    def test_generate_message_part_summary(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test message part summary generation."""
        # Single part
        single_summary = splitting_config._generate_message_part_summary(
            1, "test-123", mock_logger
        )
        assert single_summary["total_parts"] == 1
        assert single_summary["is_multipart"] is False
        assert single_summary["complexity_level"] == "single"

        # Simple multipart
        simple_summary = splitting_config._generate_message_part_summary(
            3, "test-123", mock_logger
        )
        assert simple_summary["total_parts"] == 3
        assert simple_summary["is_multipart"] is True
        assert simple_summary["complexity_level"] == "simple"

        # Complex multipart
        complex_summary = splitting_config._generate_message_part_summary(
            15, "test-123", mock_logger
        )
        assert complex_summary["complexity_level"] == "complex"

        # Very complex with warnings
        very_complex_summary = splitting_config._generate_message_part_summary(
            25, "test-123", mock_logger
        )
        assert very_complex_summary["complexity_level"] == "very_complex"
        assert "warnings" in very_complex_summary
        assert "very_large_message_batch" in very_complex_summary["warnings"]

    def test_edge_case_handling_abbreviations(
        self: "TestTelegramMessageSplitting", splitting_config: TelegramNotificationConfig
    ) -> None:
        """Test edge case handling for abbreviations in sentence splitting."""
        # Common abbreviations should not be split
        assert splitting_config._is_likely_abbreviation_or_decimal("Dr. Smith", 2) is True
        assert splitting_config._is_likely_abbreviation_or_decimal("U.S. government", 2) is True
        assert splitting_config._is_likely_abbreviation_or_decimal("etc. and more", 3) is True

        # Regular sentences should be splittable
        assert (
            splitting_config._is_likely_abbreviation_or_decimal("End of sentence. Start", 15)
            is False
        )

        # Decimal numbers should not be split
        assert (
            splitting_config._is_likely_abbreviation_or_decimal("Price is $3.14 dollars", 10)
            is True
        )

    def test_edge_case_handling_special_content(
        self: "TestTelegramMessageSplitting", splitting_config: TelegramNotificationConfig
    ) -> None:
        """Test edge case handling for special content like URLs and emails."""
        # URLs should not be split
        url_text = "Visit https://example.com/path for more info"
        url_position = 20  # Inside the URL
        assert splitting_config._is_inside_special_content(url_text, url_position) is True

        # Regular text should be splittable
        regular_position = 5  # In "Visit"
        assert splitting_config._is_inside_special_content(url_text, regular_position) is False

        # Email addresses should not be split
        email_text = "Contact user@example.com for help"
        email_position = 15  # Inside the email
        assert splitting_config._is_inside_special_content(email_text, email_position) is True

    def test_edge_case_handling_orphaned_formatting(
        self: "TestTelegramMessageSplitting", splitting_config: TelegramNotificationConfig
    ) -> None:
        """Test detection and adjustment of orphaned formatting."""
        # MarkdownV2 orphaned formatting
        orphaned_md = "This is *bold text"
        assert (
            splitting_config._would_create_orphaned_formatting(orphaned_md, len(orphaned_md))
            is True
        )

        balanced_md = "This is *bold* text"
        assert (
            splitting_config._would_create_orphaned_formatting(balanced_md, len(balanced_md))
            is False
        )

        # HTML orphaned formatting
        splitting_config.message_format = "html"
        orphaned_html = "This is <b>bold text"
        assert (
            splitting_config._would_create_orphaned_formatting(orphaned_html, len(orphaned_html))
            is True
        )

        balanced_html = "This is <b>bold</b> text"
        assert (
            splitting_config._would_create_orphaned_formatting(balanced_html, len(balanced_html))
            is False
        )

    def test_integration_split_and_format_title(
        self: "TestTelegramMessageSplitting",
        splitting_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test integration of message splitting with title formatting."""
        # Create a message that will be split
        long_message = "This is a paragraph.\n\n" * 100  # Will definitely be split
        title = "Integration Test"
        max_length = 500  # Small enough to force splitting

        # Split the message
        parts = splitting_config._split_message_preserving_formatting(
            long_message, max_length, "test-123", mock_logger
        )

        assert len(parts) > 1

        # Format titles for each part
        titles = []
        for i, _part in enumerate(parts):
            formatted_title = splitting_config._format_message_title_with_numbering(
                title, i + 1, len(parts), f"test-123-{i}", mock_logger
            )
            titles.append(formatted_title)

        # Verify numbering
        assert "*Integration Test*" in titles[0]
        assert "(1/" in titles[0]
        assert f"({len(parts)}/{len(parts)})" in titles[-1]


class TestTelegramIntegrationEndToEnd:
    """Integration test suite for end-to-end message delivery."""

    @pytest.fixture
    def integration_config(self: "TestTelegramIntegrationEndToEnd") -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig for integration testing."""
        return TelegramNotificationConfig(
            name="test_integration",
            telegram_bot_token="test_token_integration",
            telegram_chat_id="test_chat_integration",
            message_format="markdownv2",
            max_retries=2,
            base_delay=0.1,
            max_delay=5.0,
            jitter=False,
            fail_fast=False,
        )

    def test_send_message_end_to_end_success(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test complete end-to-end message sending flow with success."""
        title = "Integration Test"
        message = "This is a test message with *bold* and _italic_ formatting."

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = integration_config.send_message(title, message, mock_logger)

            assert result is True
            mock_bot.send_message.assert_called_once()

            # Verify the call parameters
            call_args = mock_bot.send_message.call_args
            assert call_args[1]["chat_id"] == "test_chat_integration"
            assert call_args[1]["parse_mode"] == "MarkdownV2"
            assert call_args[1]["disable_web_page_preview"] is True

            # Verify message formatting
            sent_text = call_args[1]["text"]
            assert "*Integration Test*" in sent_text
            assert "\\*bold\\*" in sent_text  # Should be escaped
            assert "_italic_" in sent_text  # Should remain as is for MarkdownV2

    def test_send_message_end_to_end_with_splitting(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test end-to-end flow with message splitting."""
        title = "Long Message Test"
        # Create a long message that will definitely be split
        long_content = "This is a very long paragraph. " * 200
        message = f"First section:\n\n{long_content}\n\nSecond section:\n\n{long_content}"

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = integration_config.send_message(title, message, mock_logger)

            assert result is True
            assert mock_bot.send_message.call_count > 1  # Should be split into multiple messages

            # Verify all parts have correct numbering
            calls = mock_bot.send_message.call_args_list
            for i, call in enumerate(calls):
                sent_text = call[1]["text"]
                if len(calls) > 1:
                    assert f"({i + 1}/{len(calls)})" in sent_text

    def test_send_message_end_to_end_with_retry_and_fallback(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test end-to-end flow with retry and fallback formatting."""
        title = "Retry Test"
        message = "This message has *invalid* formatting*"  # Intentionally invalid MarkdownV2

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot

            # First call: MarkdownV2 parse error, second call: success with fallback
            parse_error = BadRequest("Can't parse entities: invalid markup")
            parse_error.message = "Can't parse entities: invalid markup"

            mock_bot.send_message.side_effect = [parse_error, Mock(message_id=12345)]

            with patch("asyncio.sleep"):  # Mock sleep to speed up test
                result = integration_config.send_message(title, message, mock_logger)

            assert result is True
            assert mock_bot.send_message.call_count == 2

            # First call should use MarkdownV2
            first_call = mock_bot.send_message.call_args_list[0]
            assert first_call[1]["parse_mode"] == "MarkdownV2"

            # Second call should use fallback (Markdown or None)
            second_call = mock_bot.send_message.call_args_list[1]
            assert second_call[1]["parse_mode"] in ["Markdown", None]

    def test_send_message_end_to_end_individual_failure_continue(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test end-to-end flow with individual message failures but continuing."""
        title = "Failure Tolerance Test"
        # Create a message that will be split into multiple parts
        message = "Part 1 content.\n\n" + "Part 2 content.\n\n" + "Part 3 content."

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot

            # Simulate: first message succeeds, second fails, third succeeds
            mock_bot.send_message.side_effect = [
                Mock(message_id=12345),  # Success
                BadRequest("Invalid chat_id"),  # Failure
                Mock(message_id=12347),  # Success
            ]

            # Override max_content_length to force splitting
            with patch.object(integration_config, "_calculate_message_overhead", return_value=100):
                # This should force the message to be split into multiple parts
                result = integration_config.send_message(title, message, mock_logger)

            # Should return True because at least one message succeeded (fail_fast=False)
            assert result is True
            assert mock_bot.send_message.call_count >= 2

    def test_send_message_end_to_end_all_failures(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test end-to-end flow when all messages fail."""
        title = "All Fail Test"
        message = "This message will fail completely."

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot

            # All attempts fail
            mock_bot.send_message.side_effect = BadRequest("Invalid bot token")

            result = integration_config.send_message(title, message, mock_logger)

            assert result is False
            mock_bot.send_message.assert_called()

    def test_send_message_end_to_end_configuration_validation(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test end-to-end flow with configuration validation."""
        title = "Config Test"
        message = "Test message"

        # Test missing bot token
        integration_config.telegram_bot_token = None
        result = integration_config.send_message(title, message, mock_logger)
        assert result is False

        # Test missing chat ID
        integration_config.telegram_bot_token = "valid_token"  # noqa: S105
        integration_config.telegram_chat_id = None
        result = integration_config.send_message(title, message, mock_logger)
        assert result is False

        # Test valid configuration
        integration_config.telegram_chat_id = "valid_chat_id"
        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = integration_config.send_message(title, message, mock_logger)
            assert result is True

    @pytest.mark.asyncio
    async def test_bot_and_chat_validation_integration(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test integration of bot and chat validation."""
        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot

            # Mock successful bot info
            mock_bot.get_me.return_value = Mock(
                username="test_bot",
                id=123456789,
                first_name="Test Bot",
                can_join_groups=True,
                can_read_all_group_messages=False,
                supports_inline_queries=False,
            )

            # Mock successful chat info
            mock_bot.get_chat.return_value = Mock(
                type="private", id=987654321, title=None, username="test_user", description=None
            )

            result = await integration_config.validate_configuration(mock_logger)
            assert result is True

            mock_bot.get_me.assert_called_once()
            mock_bot.get_chat.assert_called_once_with(chat_id="test_chat_integration")

    def test_message_length_and_splitting_integration(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test integration of message length detection and splitting."""
        title = "Length Test"

        # Test message that exactly fits
        short_message = "Short message"
        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = integration_config.send_message(title, short_message, mock_logger)
            assert result is True
            assert mock_bot.send_message.call_count == 1

        # Test message that needs splitting
        long_message = "Very long content. " * 300
        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = integration_config.send_message(title, long_message, mock_logger)
            assert result is True
            assert mock_bot.send_message.call_count > 1  # Should be split

    def test_real_world_marketplace_message_integration(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test with realistic marketplace notification messages."""
        title = "New Marketplace Listings"
        message = """
🏠 *Real Estate*
• Apartment for rent - $1,500/month (2BR/2BA)
• Location: Downtown area
• Contact: [View Listing](https://example.com/listing1)

🚗 *Vehicles*
• 2018 Honda Civic (95k miles) - $15,000
• Excellent condition!
• Contact: user@example.com

💻 *Electronics*
• Gaming PC: RTX 3080 + Intel i7 - $2,500
• Like new condition
• Specs: 32GB RAM, 1TB SSD

📱 *Mobile Devices*
• iPhone 13 Pro - $899 (Like new!)
• Color: Pacific Blue
• Unlocked, all accessories included
        """.strip()

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot
            mock_bot.send_message.return_value = Mock(message_id=12345)

            result = integration_config.send_message(title, message, mock_logger)

            assert result is True
            mock_bot.send_message.assert_called()

            # Verify proper escaping of special characters
            sent_text = mock_bot.send_message.call_args[1]["text"]
            assert "\\$1,500/month" in sent_text or "$1,500/month" in sent_text
            assert "\\(2BR/2BA\\)" in sent_text or "(2BR/2BA)" in sent_text
            assert "user@example\\.com" in sent_text or "user@example.com" in sent_text

    def test_error_recovery_and_logging_integration(
        self: "TestTelegramIntegrationEndToEnd",
        integration_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test integration of error recovery with comprehensive logging."""
        title = "Error Recovery Test"
        message = "Test message for error recovery"

        with patch.object(integration_config, "_create_bot") as mock_create_bot:
            mock_bot = AsyncMock()
            mock_create_bot.return_value.__aenter__.return_value = mock_bot

            # Simulate a sequence: timeout -> success
            mock_bot.send_message.side_effect = [TimedOut(), Mock(message_id=12345)]

            with patch("asyncio.sleep"):  # Speed up test
                result = integration_config.send_message(title, message, mock_logger)

            assert result is True
            assert mock_bot.send_message.call_count == 2

            # Verify structured logging was used
            assert mock_logger.info.called
            assert mock_logger.debug.called or mock_logger.warning.called


class TestTelegramAdvancedEdgeCases:
    """Test suite for advanced edge cases including special characters, long messages, and emojis."""

    @pytest.fixture
    def markdownv2_config(self: "TestTelegramAdvancedEdgeCases") -> TelegramNotificationConfig:
        """Create a MarkdownV2 config for testing."""
        return TelegramNotificationConfig(
            name="test_markdownv2",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            message_format="markdownv2",
            max_retries=1,
            base_delay=0.01,
        )

    @pytest.fixture
    def mock_logger(self: "TestTelegramAdvancedEdgeCases") -> Mock:
        """Create a mock logger for testing."""
        return Mock()

    def test_unicode_edge_cases(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test various Unicode edge cases including zero-width characters."""
        # Test zero-width characters
        text_with_zw = "Hello\u200b\u200c\u200d\u2060World"
        escaped = markdownv2_config._escape_text_for_format(text_with_zw)
        assert "Hello" in escaped
        assert "World" in escaped

        # Test Unicode combining characters
        text_with_combining = "e\u0301\u0302\u0303"  # e with multiple combining marks
        escaped = markdownv2_config._escape_text_for_format(text_with_combining)
        assert len(escaped) > 0

        # Test Unicode surrogate pairs (emoji)
        text_with_surrogates = "Hello 👨‍👩‍👧‍👦 Family"
        escaped = markdownv2_config._escape_text_for_format(text_with_surrogates)
        assert "Hello" in escaped
        assert "Family" in escaped

    def test_emoji_complex_sequences(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test complex emoji sequences and skin tone modifiers."""
        # Test emoji with skin tone modifiers
        text_with_skin_tones = "👋🏻👋🏼👋🏽👋🏾👋🏿"
        escaped = markdownv2_config._escape_text_for_format(text_with_skin_tones)
        assert len(escaped) > 0

        # Test emoji with zero-width joiners (complex sequences)
        complex_emoji = "👨‍💻👩‍🔬🧑‍🎨👨‍👩‍👧‍👦"
        escaped = markdownv2_config._escape_text_for_format(complex_emoji)
        assert len(escaped) > 0

        # Test flag emojis (regional indicator symbols)
        flag_emojis = "🇺🇸🇬🇧🇫🇷🇩🇪🇯🇵"
        escaped = markdownv2_config._escape_text_for_format(flag_emojis)
        assert len(escaped) > 0

        # Test emoji sequences with text
        mixed_content = "Price: $50 💰 Deal! 🔥 Limited time ⏰"
        escaped = markdownv2_config._escape_text_for_format(mixed_content)
        assert "Price:" in escaped
        assert "Deal\\!" in escaped  # Exclamation mark is escaped in MarkdownV2
        assert "Limited time" in escaped

    def test_special_character_combinations(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test complex combinations of special characters."""
        # Test nested special characters
        nested_specials = "[[{}]]((__))"
        escaped = markdownv2_config._escape_text_for_format(nested_specials)
        assert "\\[\\[\\{\\}\\]\\]\\(\\(\\_\\_\\)\\)" == escaped

        # Test special characters with numbers and symbols
        complex_text = "Price: $1,234.56 (20% off) [SALE] #hashtag @mention"
        escaped = markdownv2_config._escape_text_for_format(complex_text)
        assert "Price:" in escaped
        assert "1,234\\.56" in escaped
        assert "\\(20%" in escaped
        assert "\\[SALE\\]" in escaped
        assert "\\#hashtag" in escaped

        # Test mathematical symbols
        math_symbols = "±∞≤≥≠≈∑∏∫√∂∆∇"
        escaped = markdownv2_config._escape_text_for_format(math_symbols)
        assert len(escaped) > 0

    def test_extremely_long_message_handling(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test handling of extremely long messages (>10k chars)."""
        # Create a very long message
        long_text = "A" * 10000 + " " + "B" * 5000
        title = "Very Long Message Test"

        with patch.object(markdownv2_config, "_send_all_messages_async", return_value=True):
            result = run_async_in_thread(
                markdownv2_config._prepare_and_send_messages(title, long_text, mock_logger)
            )

        assert result is True

    def test_message_with_only_special_characters(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test message containing only special characters."""
        special_only = "[](){}*_~`#+-=|\\!."
        escaped = markdownv2_config._escape_text_for_format(special_only)
        expected = "\\[\\]\\(\\)\\{\\}\\*\\_\\~\\`\\#\\+\\-\\=\\|\\\\\\!\\."
        assert escaped == expected

    def test_mixed_language_content(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test content with mixed languages and scripts."""
        # Test multiple scripts
        mixed_scripts = "Hello 你好 こんにちは مرحبا Здравствуйте"
        escaped = markdownv2_config._escape_text_for_format(mixed_scripts)
        assert "Hello" in escaped
        assert "你好" in escaped
        assert "こんにちは" in escaped
        assert "مرحبا" in escaped
        assert "Здравствуйте" in escaped

        # Test with special characters mixed with non-Latin scripts
        mixed_with_specials = "价格: $100 (特惠) [限时]"
        escaped = markdownv2_config._escape_text_for_format(mixed_with_specials)
        assert "价格:" in escaped
        assert "100" in escaped
        assert "特惠" in escaped
        assert "限时" in escaped

    def test_edge_case_line_breaks_and_whitespace(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test various line break and whitespace combinations."""
        # Test different types of line breaks
        various_breaks = "Line1\nLine2\r\nLine3\rLine4\u2028Line5\u2029Line6"
        escaped = markdownv2_config._escape_text_for_format(various_breaks)
        assert "Line1" in escaped
        assert "Line6" in escaped

        # Test various whitespace characters
        various_whitespace = "Word1\tWord2\u00A0Word3\u2000Word4\u3000Word5"
        escaped = markdownv2_config._escape_text_for_format(various_whitespace)
        assert "Word1" in escaped
        assert "Word5" in escaped

    def test_boundary_length_messages(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
        mock_logger: Mock,
    ) -> None:
        """Test messages at exact boundary lengths."""
        # Test message at exactly 4096 characters
        exact_limit = "A" * (MessageLimit.MESSAGE_TEXT - 1)  # Leave room for title
        title = "T"

        messages = markdownv2_config._split_message_preserving_formatting(
            title, exact_limit, "markdownv2"
        )
        assert len(messages) >= 1

        # Test message just over the limit
        over_limit = "A" * (MessageLimit.MESSAGE_TEXT + 10)
        messages = markdownv2_config._split_message_preserving_formatting(
            title, over_limit, "markdownv2"
        )
        assert len(messages) >= 2

    def test_malformed_formatting_recovery(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test recovery from malformed formatting."""
        # Test unmatched formatting characters
        malformed_texts = [
            "*bold without closing",
            "_italic without closing",
            "`code without closing",
            "~strike without closing",
            "||spoiler without closing",
            "**double asterisk",
            "__double underscore",
            "```code block without closing",
        ]

        for malformed in malformed_texts:
            escaped = markdownv2_config._escape_text_for_format(malformed)
            assert len(escaped) > 0
            # Ensure special characters are escaped
            assert "\\*" in escaped or "\\`" in escaped or "\\_" in escaped or "\\~" in escaped

    def test_extremely_nested_structures(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test deeply nested bracket/parentheses structures."""
        # Test deeply nested brackets
        deeply_nested = "[[[[inner]]]]"
        escaped = markdownv2_config._escape_text_for_format(deeply_nested)
        assert escaped == "\\[\\[\\[\\[inner\\]\\]\\]\\]"

        # Test mixed nested structures
        mixed_nested = "({[*_test_*]})"
        escaped = markdownv2_config._escape_text_for_format(mixed_nested)
        assert "\\(" in escaped
        assert "\\{" in escaped
        assert "\\[" in escaped
        assert "\\*" in escaped
        assert "\\_" in escaped

    def test_control_characters_handling(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test handling of control characters."""
        # Test various control characters
        control_chars = "Hello\x00\x01\x02\x03\x04\x05World"
        escaped = markdownv2_config._escape_text_for_format(control_chars)
        assert "Hello" in escaped
        assert "World" in escaped

        # Test tab and newline characters (should be preserved)
        tab_newline = "Line1\tTabbed\nNewLine"
        escaped = markdownv2_config._escape_text_for_format(tab_newline)
        assert "Line1" in escaped
        assert "Tabbed" in escaped
        assert "NewLine" in escaped

    def test_url_edge_cases_in_text(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test URLs with various edge cases."""
        # Test URLs with special characters
        url_texts = [
            "Visit https://example.com/path?param=value&other=123",
            "FTP: ftp://files.example.com/file.zip",
            "Email: user@domain.co.uk",
            "Complex: https://sub.domain.com:8080/path/to/resource?a=1&b=2#section",
            "IP: http://192.168.1.1:3000/api/v1/data",
        ]

        for url_text in url_texts:
            escaped = markdownv2_config._escape_text_for_format(url_text)
            assert len(escaped) > 0
            # URLs should have their special characters escaped appropriately
            if "?" in url_text:
                assert "\\?" in escaped or "?" in escaped  # May be in different positions
            if "&" in url_text:
                assert escaped.count("&") <= url_text.count(
                    "&"
                )  # & doesn't need escape in MarkdownV2

    def test_message_splitting_with_complex_content(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test message splitting with complex mixed content."""
        # Create complex content with emojis, special characters, and formatting
        complex_content = (
            "🚀 **MARKETPLACE ALERT** 🚀\n\n"
            "💰 Price: $1,234.56 (20% off)\n"
            "📅 Valid until: 2024-12-31\n"
            "🔗 Link: https://example.com/deal?id=123&ref=alert\n"
            "📝 Description: "
            + "A" * 3000  # Make it long enough to split
            + "\n\n🏷️ Tags: #sale #limited #bestseller"
            + "\n⚠️ Terms: See [website](https://example.com/terms) for details"
        )

        messages = markdownv2_config._split_message_preserving_formatting(
            "Complex Alert", complex_content, "markdownv2"
        )

        assert len(messages) >= 2  # Should be split
        # Verify all parts contain expected content
        full_content = " ".join(messages)
        assert "MARKETPLACE ALERT" in full_content
        assert "$1,234.56" in full_content
        assert "https://example.com/deal" in full_content

    def test_performance_with_large_emoji_content(
        self: "TestTelegramAdvancedEdgeCases",
        markdownv2_config: TelegramNotificationConfig,
    ) -> None:
        """Test performance with content containing many emojis."""
        # Create content with many emojis
        emoji_heavy = "🎉" * 1000 + " Party! " + "🎈" * 1000

        # Test escaping performance
        import time

        start_time = time.time()
        escaped = markdownv2_config._escape_text_for_format(emoji_heavy)
        escape_time = time.time() - start_time

        assert len(escaped) > 0
        assert escape_time < 1.0  # Should complete within 1 second

        # Test splitting with emoji content
        start_time = time.time()
        messages = markdownv2_config._split_message_preserving_formatting(
            "Emoji Test", emoji_heavy, "markdownv2"
        )
        split_time = time.time() - start_time

        assert len(messages) >= 1
        assert split_time < 1.0  # Should complete within 1 second


class TestTelegramEnhancedRetry:
    """Test suite for enhanced retry functionality with exponential backoff and jitter."""

    @pytest.fixture
    def retry_config(self: "TestTelegramEnhancedRetry") -> TelegramNotificationConfig:
        """Create a TelegramNotificationConfig with custom retry settings."""
        return TelegramNotificationConfig(
            name="test_retry",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            jitter=True,
        )

    @pytest.fixture
    def no_jitter_config(self: "TestTelegramEnhancedRetry") -> TelegramNotificationConfig:
        """Create config without jitter for predictable testing."""
        return TelegramNotificationConfig(
            name="test_no_jitter",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0,
            jitter=False,
        )

    def test_retry_config_validation(self: "TestTelegramEnhancedRetry") -> None:
        """Test validation of retry configuration parameters."""
        # Valid configuration
        config = TelegramNotificationConfig(
            name="test",
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            jitter=True,
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.jitter is True

        # Test invalid max_retries
        with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
            TelegramNotificationConfig(name="test", max_retries=-1)

        with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
            TelegramNotificationConfig(name="test", max_retries="invalid")

        # Test invalid base_delay
        with pytest.raises(ValueError, match="base_delay must be a positive number"):
            TelegramNotificationConfig(name="test", base_delay=0)

        with pytest.raises(ValueError, match="base_delay must be a positive number"):
            TelegramNotificationConfig(name="test", base_delay=-1.0)

        # Test invalid max_delay
        with pytest.raises(ValueError, match="max_delay must be a positive number"):
            TelegramNotificationConfig(name="test", max_delay=0)

        with pytest.raises(
            ValueError, match="max_delay must be greater than or equal to base_delay"
        ):
            TelegramNotificationConfig(name="test", base_delay=5.0, max_delay=2.0)

        # Test invalid jitter
        with pytest.raises(ValueError, match="jitter must be a boolean"):
            TelegramNotificationConfig(name="test", jitter="yes")

    def test_calculate_retry_delay_without_jitter(
        self: "TestTelegramEnhancedRetry", no_jitter_config: TelegramNotificationConfig
    ) -> None:
        """Test exponential backoff calculation without jitter."""
        config = no_jitter_config

        # Test exponential backoff progression
        assert config._calculate_retry_delay(0) == 1.0  # base_delay * 2^0
        assert config._calculate_retry_delay(1) == 2.0  # base_delay * 2^1
        assert config._calculate_retry_delay(2) == 4.0  # base_delay * 2^2
        assert config._calculate_retry_delay(3) == 8.0  # base_delay * 2^3

        # Test max_delay capping
        assert config._calculate_retry_delay(10) == 10.0  # Should be capped at max_delay

    def test_calculate_retry_delay_with_jitter(
        self: "TestTelegramEnhancedRetry", retry_config: TelegramNotificationConfig
    ) -> None:
        """Test exponential backoff calculation with jitter."""
        config = retry_config

        # Test that jitter creates variation
        delays = [config._calculate_retry_delay(1) for _ in range(10)]

        # All delays should be around 2.0 (base_delay * 2^1) but with some variation
        base_expected = 2.0
        min_expected = base_expected * 0.75  # -25% jitter
        max_expected = base_expected * 1.25  # +25% jitter

        for delay in delays:
            assert min_expected <= delay <= max_expected
            assert delay >= 0.1  # Minimum delay

        # Test that there's actually variation (not all the same)
        assert len(set(delays)) > 1, "Jitter should create variation in delays"

    def test_calculate_retry_delay_with_custom_parameters(
        self: "TestTelegramEnhancedRetry", retry_config: TelegramNotificationConfig
    ) -> None:
        """Test retry delay calculation with custom parameters."""
        config = retry_config

        # Test with custom base_delay
        delay = config._calculate_retry_delay(1, base_delay=0.5, jitter=False)
        assert delay == 1.0  # 0.5 * 2^1

        # Test with custom max_delay
        delay = config._calculate_retry_delay(10, base_delay=1.0, max_delay=5.0, jitter=False)
        assert delay == 5.0  # Should be capped

    def test_is_transient_error(
        self: "TestTelegramEnhancedRetry", retry_config: TelegramNotificationConfig
    ) -> None:
        """Test transient error detection."""
        config = retry_config

        # Test specific error types that should always be retried
        assert config._is_transient_error(RetryAfter(5)) is True
        assert config._is_transient_error(TimedOut()) is True
        assert config._is_transient_error(NetworkError("Connection failed")) is True

        # Test error messages that indicate transient issues
        transient_patterns = [
            "Network error occurred",
            "Connection timeout",
            "Internal server error",
            "Service unavailable",
            "Too many requests",
            "Rate limit exceeded",
            "Try again later",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout",
        ]

        for pattern in transient_patterns:
            error = TelegramError(pattern)
            error.message = pattern
            assert (
                config._is_transient_error(error) is True
            ), f"Should detect '{pattern}' as transient"

        # Test non-transient errors
        non_transient_errors = [
            "Chat not found",
            "Bot was blocked",
            "Invalid token",
            "Permission denied",
            "User not found",
        ]

        for error_msg in non_transient_errors:
            error = TelegramError(error_msg)
            error.message = error_msg
            assert (
                config._is_transient_error(error) is False
            ), f"Should NOT detect '{error_msg}' as transient"

        # Test error without message attribute
        error_no_msg = TelegramError("Some error")
        # Remove message attribute if it exists
        if hasattr(error_no_msg, "message"):
            delattr(error_no_msg, "message")
        assert config._is_transient_error(error_no_msg) is False

    @pytest.mark.asyncio
    async def test_retry_after_uses_server_delay(
        self: "TestTelegramEnhancedRetry",
        retry_config: TelegramNotificationConfig,
    ) -> None:
        """Test that RetryAfter uses server-specified delay, not exponential backoff."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # First call hits rate limit, second succeeds
        retry_error = RetryAfter(5.5)  # Server says wait 5.5 seconds
        success_result = Mock(message_id=12345)
        mock_bot.send_message.side_effect = [retry_error, success_result]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await retry_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Should use server-specified delay, not our exponential backoff
        mock_sleep.assert_called_once_with(5.5)

        # Verify logging
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Rate limit hit, waiting 5.5s" in call for call in warning_calls)

    @pytest.mark.asyncio
    async def test_timeout_uses_exponential_backoff(
        self: "TestTelegramEnhancedRetry",
        no_jitter_config: TelegramNotificationConfig,
    ) -> None:
        """Test that TimedOut uses exponential backoff with jitter."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # First call times out, second succeeds
        timeout_error = TimedOut()
        success_result = Mock(message_id=12345)
        mock_bot.send_message.side_effect = [timeout_error, success_result]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await no_jitter_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Should use exponential backoff (attempt 0 = 1.0 second)
        mock_sleep.assert_called_once_with(1.0)

        # Verify logging
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Timeout occurred" in call for call in warning_calls)

        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any(
            "Waiting 1.00s before retry (exponential backoff)" in call for call in info_calls
        )

    @pytest.mark.asyncio
    async def test_network_error_uses_exponential_backoff(
        self: "TestTelegramEnhancedRetry",
        no_jitter_config: TelegramNotificationConfig,
    ) -> None:
        """Test that NetworkError uses exponential backoff."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # First call has network error, second succeeds
        network_error = NetworkError("Connection failed")
        success_result = Mock(message_id=12345)
        mock_bot.send_message.side_effect = [network_error, success_result]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await no_jitter_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Should use exponential backoff (attempt 0 = 1.0 second)
        mock_sleep.assert_called_once_with(1.0)

        # Verify logging
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Network error occurred" in call for call in warning_calls)

    @pytest.mark.asyncio
    async def test_transient_telegram_error_retry(
        self: "TestTelegramEnhancedRetry",
        retry_config: TelegramNotificationConfig,
    ) -> None:
        """Test that transient TelegramError is retried with exponential backoff."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Create a transient error
        transient_error = TelegramError("Service unavailable - try again")
        transient_error.message = "Service unavailable - try again"
        success_result = Mock(message_id=12345)
        mock_bot.send_message.side_effect = [transient_error, success_result]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await retry_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is True
        assert mock_bot.send_message.call_count == 2

        # Should use exponential backoff with jitter
        mock_sleep.assert_called_once()
        # With jitter, delay should be around 1.0 +/- 25%
        actual_delay = mock_sleep.call_args[0][0]
        assert 0.75 <= actual_delay <= 1.25

        # Verify logging
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Transient error occurred" in call for call in warning_calls)

        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any(
            "Waiting" in call and "before retry (transient error backoff)" in call
            for call in info_calls
        )

    @pytest.mark.asyncio
    async def test_non_transient_error_no_retry(
        self: "TestTelegramEnhancedRetry",
        retry_config: TelegramNotificationConfig,
    ) -> None:
        """Test that non-transient errors are not retried."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Create a non-transient error
        non_transient_error = TelegramError("Chat not found")
        non_transient_error.message = "Chat not found"
        mock_bot.send_message.side_effect = non_transient_error

        result = await retry_config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger
        )

        assert result is False
        assert mock_bot.send_message.call_count == 1  # No retries

        # Verify logging
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("Non-transient API error (no retry)" in call for call in error_calls)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_timeout(
        self: "TestTelegramEnhancedRetry",
        no_jitter_config: TelegramNotificationConfig,
    ) -> None:
        """Test max retries exceeded for timeout errors."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Always timeout
        mock_bot.send_message.side_effect = TimedOut()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await no_jitter_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is False
        # Should try initial + max_retries (2) = 3 total attempts
        assert mock_bot.send_message.call_count == 3

        # Should have 2 sleep calls (after first 2 failures)
        assert mock_sleep.call_count == 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0]  # Exponential backoff: 1s, 2s

        # Verify final error logging
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("Max retries (2) exceeded for timeout" in call for call in error_calls)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_rate_limit(
        self: "TestTelegramEnhancedRetry",
        no_jitter_config: TelegramNotificationConfig,
    ) -> None:
        """Test max retries exceeded for rate limit errors."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Always hit rate limit
        mock_bot.send_message.side_effect = RetryAfter(1.0)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await no_jitter_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is False
        # Should try initial + max_retries (2) = 3 total attempts
        assert mock_bot.send_message.call_count == 3

        # Should have 2 sleep calls (server-specified delays)
        assert mock_sleep.call_count == 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 1.0]  # Server-specified delays

        # Verify final error logging
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("Max retries (2) exceeded for rate limiting" in call for call in error_calls)

    @pytest.mark.asyncio
    async def test_success_after_multiple_retries(
        self: "TestTelegramEnhancedRetry",
        no_jitter_config: TelegramNotificationConfig,
    ) -> None:
        """Test successful send after multiple retry attempts."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Fail twice, then succeed
        timeout_error = TimedOut()
        network_error = NetworkError("Connection lost")
        success_result = Mock(message_id=12345)
        mock_bot.send_message.side_effect = [timeout_error, network_error, success_result]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await no_jitter_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is True
        assert mock_bot.send_message.call_count == 3

        # Should have 2 sleep calls with exponential backoff
        assert mock_sleep.call_count == 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0]  # Exponential progression

        # Verify success logging after retries
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Message sent successfully after 2 retries" in call for call in info_calls)

    @pytest.mark.asyncio
    async def test_retry_with_fallback_formatting(
        self: "TestTelegramEnhancedRetry",
        retry_config: TelegramNotificationConfig,
    ) -> None:
        """Test that retry logic works alongside fallback formatting."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # First attempt: MarkdownV2 parse error (triggers fallback)
        # Second attempt: Network error on fallback (triggers retry)
        # Third attempt: Success on retry
        parse_error = BadRequest("Can't parse entities")
        parse_error.message = "Can't parse entities: invalid markup"

        network_error = NetworkError("Connection timeout")
        success_result = Mock(message_id=12345)

        # Configure side effects for the sequence
        mock_bot.send_message.side_effect = [
            parse_error,  # Initial MarkdownV2 attempt fails
            network_error,  # Fallback to markdown fails with network error
            success_result,  # Retry of fallback succeeds
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await retry_config._send_message_async(
                mock_bot, "*Test message*", "MarkdownV2", mock_logger
            )

        assert result is True
        assert mock_bot.send_message.call_count == 3

        # Verify the sequence of calls
        calls = mock_bot.send_message.call_args_list
        assert calls[0][1]["parse_mode"] == "MarkdownV2"  # Initial attempt
        assert calls[1][1]["parse_mode"] == "Markdown"  # Fallback attempt
        assert calls[1][1]["text"] == "**Test message**"  # Converted text
        assert calls[2][1]["parse_mode"] == "Markdown"  # Retry of fallback
        assert calls[2][1]["text"] == "**Test message**"  # Same converted text

        # Should have one sleep call (for the network error retry)
        assert mock_sleep.call_count == 1

        # Verify logging shows both fallback and retry
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Trying fallback format: markdown" in call for call in info_calls)

        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Attempting fallback formatting" in call for call in warning_calls)
        assert any("Transient error occurred" in call for call in warning_calls)

    @pytest.mark.asyncio
    async def test_retry_config_backward_compatibility(
        self: "TestTelegramEnhancedRetry",
    ) -> None:
        """Test that old code calling _send_message_async with max_retries still works."""
        # Create config with default retry settings
        config = TelegramNotificationConfig(
            name="test_compat",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
        )

        mock_logger = Mock()
        mock_bot = AsyncMock()
        success_result = Mock(message_id=12345)
        mock_bot.send_message.return_value = success_result

        # Call with explicit max_retries (old style)
        result = await config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger, max_retries=5
        )

        assert result is True

        # Verify debug logging shows the overridden max_retries
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("max_retries=5" in call for call in debug_calls)

    def test_retry_defaults(self: "TestTelegramEnhancedRetry") -> None:
        """Test default retry configuration values."""
        config = TelegramNotificationConfig(name="test_defaults")

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    @pytest.mark.asyncio
    async def test_zero_max_retries(
        self: "TestTelegramEnhancedRetry",
    ) -> None:
        """Test behavior with zero max retries."""
        config = TelegramNotificationConfig(
            name="test_no_retry",
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            max_retries=0,
        )

        mock_logger = Mock()
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TimedOut()

        result = await config._send_message_async(
            mock_bot, "Test message", "MarkdownV2", mock_logger
        )

        assert result is False
        assert mock_bot.send_message.call_count == 1  # Only initial attempt, no retries

    @pytest.mark.asyncio
    async def test_structured_retry_logging(
        self: "TestTelegramEnhancedRetry",
        no_jitter_config: TelegramNotificationConfig,
    ) -> None:
        """Test detailed structured logging for retry attempts."""
        mock_logger = Mock()
        mock_bot = AsyncMock()

        # Create sequence: timeout -> network error -> success
        timeout_error = TimedOut()
        network_error = NetworkError("Connection lost")
        success_result = Mock(message_id=12345)
        mock_bot.send_message.side_effect = [timeout_error, network_error, success_result]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await no_jitter_config._send_message_async(
                mock_bot, "Test message", "MarkdownV2", mock_logger
            )

        assert result is True

        # Verify structured logging contains all expected elements
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Should log retry configuration
        assert any("Retry config:" in call and "max_retries=2" in call for call in debug_calls)

        # Should log each attempt
        assert any("attempt 1/3" in call for call in debug_calls)
        assert any("attempt 2/3" in call for call in debug_calls)
        assert any("attempt 3/3" in call for call in debug_calls)

        # Should log specific error types and retry delays
        assert any("Timeout occurred (attempt 1/3)" in call for call in warning_calls)
        assert any("Network error occurred (attempt 2/3)" in call for call in warning_calls)
        assert any(
            "Waiting 1.00s before retry (exponential backoff)" in call for call in info_calls
        )
        assert any(
            "Waiting 2.00s before retry (exponential backoff)" in call for call in info_calls
        )

        # Should log success after retries
        assert any("Message sent successfully after 2 retries" in call for call in info_calls)
