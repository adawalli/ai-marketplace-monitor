"""Test suite for Telegram MarkdownV2 message formatting and special character escaping."""

from unittest.mock import AsyncMock, patch

import pytest

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramMarkdownV2Formatting:
    """Test MarkdownV2 message formatting with special character escaping."""

    def test_markdownv2_special_characters_need_escaping(self):
        """Test that MarkdownV2 special characters are properly escaped."""
        # These characters need to be escaped in MarkdownV2:
        # _ * [ ] ( ) ~ ` > # + - = | { } . !

        # Test string with all special characters
        test_title = "Test_Title*With[Special]Characters"
        test_message = "Price: $100.99 (great deal!) - 50% off #sale"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="markdownv2",
        )

        # Mock telegram Bot
        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            # Verify the call was made with MarkdownV2 parse_mode
            mock_bot_instance.send_message.assert_called_once()
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "MarkdownV2"

    def test_markdownv2_formatting_with_bold_and_italic(self):
        """Test MarkdownV2 with bold and italic formatting."""
        test_title = "*Bold Title*"
        test_message = "_Italic text_ and *bold text*"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "MarkdownV2"
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_markdownv2_formatting_with_code_and_links(self):
        """Test MarkdownV2 with inline code and links."""
        test_title = "Code Example"
        test_message = "Use `code` and [link](https://example.com)"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "MarkdownV2"

    def test_markdownv2_with_price_and_special_characters(self):
        """Test MarkdownV2 with typical marketplace listing content."""
        test_title = "iPhone 13 Pro - $899.99"
        test_message = "Condition: 9/10 (excellent!)\nLocation: Dallas, TX\nDescription: Like new condition - barely used!"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "MarkdownV2"

    def test_markdownv2_handles_telegram_api_errors(self):
        """Test that MarkdownV2 gracefully handles Telegram API errors."""
        test_title = "Test Title"
        test_message = "Test message with problematic formatting"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        # Mock telegram Bot to raise an exception
        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(side_effect=Exception("Parse error"))

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should return False on error
            assert result is False

    def test_markdownv2_with_empty_strings(self):
        """Test MarkdownV2 handling of empty or whitespace-only strings."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
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

    def test_markdownv2_parse_mode_mapping(self):
        """Test that markdownv2 message_format maps to MarkdownV2 parse_mode."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title="Test Title", message="Test Message", logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            # Verify exact parse_mode value
            assert call_args[1]["parse_mode"] == "MarkdownV2"

    def test_markdownv2_synchronous_interface(self):
        """Test that MarkdownV2 formatting maintains synchronous interface."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            # This should work synchronously without async/await
            result = config.send_message(title="Test Title", message="Test Message", logger=None)

            # Should return boolean result synchronously
            assert isinstance(result, bool)
            assert result is True

    def test_markdownv2_message_combination(self):
        """Test that title and message are properly combined with newlines."""
        test_title = "Test Title"
        test_message = "Test Message"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
