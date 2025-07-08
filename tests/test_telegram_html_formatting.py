"""Test suite for Telegram HTML message formatting and tag handling."""

from unittest.mock import AsyncMock, patch

import pytest

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramHtmlFormatting:
    """Test HTML message formatting with proper tag usage and escaping."""

    def test_html_basic_formatting_tags(self):
        """Test HTML formatting with basic tags (bold, italic, code)."""
        test_title = "<b>Bold Title</b>"
        test_message = "<i>Italic text</i> and <b>bold text</b> with <code>inline code</code>"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"
            expected_text = f"{test_title}\n\n{test_message}"
            assert call_args[1]["text"] == expected_text

    def test_html_link_formatting(self):
        """Test HTML formatting with links."""
        test_title = "Link Test"
        test_message = 'Check out this <a href="https://example.com">amazing deal</a>!'

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_special_characters_escaping(self):
        """Test HTML formatting with special characters that need escaping."""
        test_title = "Price: $100 & up"
        test_message = 'Description: <Good condition> & "like new"'

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_pre_and_code_blocks(self):
        """Test HTML formatting with pre and code blocks."""
        test_title = "Code Example"
        test_message = "<pre>console.log('Hello World!');</pre> and <code>inline code</code>"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_marketplace_listing_format(self):
        """Test HTML formatting with typical marketplace listing content."""
        test_title = "<b>iPhone 13 Pro</b> - $899.99"
        test_message = (
            "<b>Condition:</b> 9/10 (excellent!)\n"
            "<b>Location:</b> Dallas, TX\n"
            "<b>Description:</b> Like new condition - <i>barely used</i>!\n"
            '<a href="https://example.com/item">View listing</a>'
        )

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_handles_telegram_api_errors(self):
        """Test that HTML formatting gracefully handles Telegram API errors."""
        test_title = "Test Title"
        test_message = "<b>Test message</b> with <invalid>HTML</invalid>"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        # Mock telegram Bot to raise an exception
        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(side_effect=Exception("Parse error"))

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should return False on error
            assert result is False

    def test_html_with_empty_strings(self):
        """Test HTML handling of empty or whitespace-only strings."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        # Test empty title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message(title="", message="Valid message", logger=None)

        # Test empty message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message(title="Valid title", message="", logger=None)

    def test_html_parse_mode_mapping(self):
        """Test that html message_format maps to HTML parse_mode."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title="Test Title", message="Test Message", logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            # Verify exact parse_mode value
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_synchronous_interface(self):
        """Test that HTML formatting maintains synchronous interface."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            # This should work synchronously without async/await
            result = config.send_message(
                title="<b>Test Title</b>", message="<i>Test Message</i>", logger=None
            )

            # Should return boolean result synchronously
            assert isinstance(result, bool)
            assert result is True

    def test_html_strikethrough_and_underline(self):
        """Test HTML formatting with strikethrough and underline."""
        test_title = "<u>Underlined Title</u>"
        test_message = "<s>Strikethrough text</s> and <u>underlined text</u>"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_nested_tags(self):
        """Test HTML formatting with nested tags."""
        test_title = "<b><i>Bold and Italic</i></b>"
        test_message = '<b>Bold with <code>inline code</code></b> and <i>italic with <a href="https://example.com">link</a></i>'

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            assert result is True
            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] == "HTML"

    def test_html_message_combination(self):
        """Test that title and message are properly combined with newlines."""
        test_title = "<b>Test Title</b>"
        test_message = "<i>Test Message</i>"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
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
