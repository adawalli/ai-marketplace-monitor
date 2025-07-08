"""Tests for Telegram retry logic and error handling."""

from unittest.mock import Mock, patch

import telegram

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramRetryLogic:
    """Test suite for retry behavior in Telegram notifications."""

    def test_successful_send_no_retry_needed(self):
        """Test that successful sends don't trigger retry logic."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:test_token",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot

            # Mock successful send
            mock_bot.send_message.return_value = True

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = True

                result = config.send_message("Test Title", "Test Message")

                assert result is True
                assert mock_run.call_count == 1  # Should only call once

    def test_network_error_triggers_retry(self):
        """Test that network errors trigger retry logic."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:test_token",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot

            # Mock network error on first call, success on second
            mock_bot.send_message.side_effect = [
                telegram.error.NetworkError("Network timeout"),
                True,
            ]

            with patch("asyncio.run") as mock_run:
                mock_run.side_effect = [telegram.error.NetworkError("Network timeout"), True]

                with patch("time.sleep") as mock_sleep:
                    result = config.send_message("Test Title", "Test Message")

                    assert result is True
                    assert mock_run.call_count == 2  # Should retry once
                    assert mock_sleep.call_count == 1  # Should sleep once

    def test_max_retries_exceeded(self):
        """Test that max retries are respected."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:test_token",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot

            # Mock persistent network error
            mock_bot.send_message.side_effect = telegram.error.NetworkError("Persistent error")

            with patch("asyncio.run") as mock_run:
                mock_run.side_effect = telegram.error.NetworkError("Persistent error")

                with patch("time.sleep") as mock_sleep:
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    assert mock_run.call_count == 5  # Should try 5 times total
                    assert mock_sleep.call_count == 4  # Should sleep 4 times

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff delays are correct."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:test_token",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot

            # Mock persistent network error
            mock_bot.send_message.side_effect = telegram.error.NetworkError("Persistent error")

            with patch("asyncio.run") as mock_run:
                mock_run.side_effect = telegram.error.NetworkError("Persistent error")

                with patch("time.sleep") as mock_sleep:
                    config.send_message("Test Title", "Test Message")

                    # Check that sleep was called with exponential backoff
                    expected_delays = [0.1, 0.2, 0.4, 0.8]
                    actual_delays = [call[0][0] for call in mock_sleep.call_args_list]

                    assert actual_delays == expected_delays

    def test_retry_after_error_respected(self):
        """Test that RetryAfter errors respect the suggested delay."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:test_token",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot

            # Mock RetryAfter error then success
            retry_after_error = telegram.error.RetryAfter(5)
            mock_bot.send_message.side_effect = [retry_after_error, True]

            with patch("asyncio.run") as mock_run:
                mock_run.side_effect = [retry_after_error, True]

                with patch("time.sleep") as mock_sleep:
                    result = config.send_message("Test Title", "Test Message")

                    assert result is True
                    assert mock_run.call_count == 2
                    assert mock_sleep.call_count == 1
                    # Should sleep for the RetryAfter duration
                    assert mock_sleep.call_args[0][0] == 5
