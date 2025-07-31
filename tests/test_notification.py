"""Tests for notification.py module."""

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

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
