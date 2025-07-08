"""Test suite for Telegram plain text message formatting and handling."""

from unittest.mock import AsyncMock, patch

import pytest

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramPlainTextFormatting:
    """Test plain text message formatting without any formatting or escaping."""

    def test_plain_text_basic_message(self):
        """Test plain text formatting with basic message content."""
        test_title = "Plain Text Title"
        test_message = "This is a plain text message with no formatting."

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None  # No parse_mode for plain text
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_plain_text_with_special_characters(self):
        """Test plain text formatting with special characters that would need escaping in other formats."""
        test_title = "Special Characters: *_[]()~`>#+-=|{}.!"
        test_message = "Price: $100.99 (great deal!) - 50% off #sale & more!"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None
            # Characters should be sent as-is without escaping
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_plain_text_with_markup_like_content(self):
        """Test plain text formatting with content that looks like markup but should be treated as plain text."""
        test_title = "<b>Not Bold</b> and *Not Italic*"
        test_message = "[Not a link](https://example.com) and `not code`"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None
            # Markup-like content should be sent as-is without interpretation
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_plain_text_marketplace_listing_content(self):
        """Test plain text formatting with typical marketplace listing content."""
        test_title = "iPhone 13 Pro - $899.99"
        test_message = (
            "Condition: 9/10 (excellent!)\n"
            "Location: Dallas, TX\n"
            "Description: Like new condition - barely used!\n"
            "Contact: Call/text 555-123-4567\n"
            "Price negotiable! Must sell by Friday."
        )

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None

    def test_plain_text_handles_telegram_api_errors(self):
        """Test that plain text formatting gracefully handles Telegram API errors."""
        test_title = "Test Title"
        test_message = "Test message"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        # Mock telegram Bot to raise an exception
        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(side_effect=Exception("Network error"))

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should return False on error
            assert result is False

    def test_plain_text_with_empty_strings(self):
        """Test plain text handling of empty or whitespace-only strings."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        # Test empty title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message(title="", message="Valid message", logger=None)

        # Test empty message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message(title="Valid title", message="", logger=None)

        # Test whitespace-only title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message(title="   ", message="Valid message", logger=None)

        # Test whitespace-only message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message(title="Valid title", message="   ", logger=None)

    def test_plain_text_parse_mode_mapping(self):
        """Test that plain_text message_format maps to None parse_mode."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title="Test Title", message="Test Message", logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            # Verify parse_mode is None for plain text
            assert call_args[1]["parse_mode"] is None

    def test_plain_text_synchronous_interface(self):
        """Test that plain text formatting maintains synchronous interface."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            # This should work synchronously without async/await
            result = config.send_message(title="Test Title", message="Test Message", logger=None)

            # Should return boolean result synchronously
            assert isinstance(result, bool)
            assert result is True

    def test_plain_text_with_unicode_characters(self):
        """Test plain text formatting with Unicode characters."""
        test_title = "🎯 Great Deal Alert! 💰"
        test_message = "Special characters: é, ñ, ü, 中文, العربية, 🚀"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None
            # Unicode should be preserved as-is
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_plain_text_message_combination(self):
        """Test that title and message are properly combined with newlines."""
        test_title = "Test Title"
        test_message = "Test Message"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_plain_text_fallback_behavior(self):
        """Test that plain text works as a fallback when other formats fail."""
        # This test verifies that plain text is a reliable fallback option
        # that doesn't require any special processing or escaping
        test_title = "Fallback Test: *[()]{}~`>#+-=|.!"
        test_message = (
            "This content would break MarkdownV2 or HTML parsing but works fine as plain text."
        )

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None
            # All content should be sent as-is without any processing
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
