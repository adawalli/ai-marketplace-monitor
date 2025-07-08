"""Test suite for special character escaping and fallback logic in Telegram formatting."""

from unittest.mock import AsyncMock, patch

import pytest

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramEscapingAndFallback:
    """Test special character escaping and format fallback mechanisms."""

    def test_markdownv2_escaping_preserves_formatting_markup(self):
        """Test that MarkdownV2 escaping preserves intentional formatting while escaping special chars."""
        # Bold text should remain bold, but other special chars should be escaped
        test_title = "*Bold Title* - Price: $99.99!"
        test_message = "_Italic text_ with special chars: [brackets] and (parentheses)"

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

            # The message should have special characters escaped but preserve formatting
            sent_text = call_args[1]["text"]

            # Should preserve bold and italic formatting
            assert "*Bold Title*" in sent_text
            assert "_Italic text_" in sent_text

            # Should escape special characters outside of formatting
            assert "\\$" in sent_text  # Dollar sign should be escaped
            assert "\\!" in sent_text  # Exclamation should be escaped
            assert "\\[" in sent_text and "\\]" in sent_text  # Brackets should be escaped
            assert "\\(" in sent_text and "\\)" in sent_text  # Parentheses should be escaped

    def test_markdownv2_escaping_complex_scenarios(self):
        """Test MarkdownV2 escaping with complex mixed content."""
        test_title = "iPhone *13 Pro* - $899.99 (Great Deal!)"
        test_message = "Features: _5G ready_, `A15 chip`, and [more info](https://example.com). Price: $899.99 + tax!"

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
            sent_text = call_args[1]["text"]

            # Should preserve formatting markup
            assert "*13 Pro*" in sent_text
            assert "_5G ready_" in sent_text
            assert "`A15 chip`" in sent_text
            # Note: Links get brackets escaped - this is correct behavior for safety
            assert "\\[more info\\]" in sent_text

            # Should escape standalone special characters
            assert "\\$" in sent_text
            assert "\\!" in sent_text
            assert "\\(" in sent_text and "\\)" in sent_text
            assert "\\+" in sent_text

    def test_html_escaping_preserves_valid_tags(self):
        """Test that HTML escaping preserves valid HTML tags while escaping content."""
        test_title = "<b>Bold Title</b> - Price: $99.99 & up"
        test_message = (
            "<i>Italic text</i> with special chars: <script>alert('test')</script> & more"
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
            sent_text = call_args[1]["text"]

            # Should preserve valid HTML formatting tags
            assert "<b>Bold Title</b>" in sent_text
            assert "<i>Italic text</i>" in sent_text

            # Should escape dangerous/invalid HTML and special characters
            assert "&amp;" in sent_text  # & should be escaped to &amp;
            assert "&lt;script&gt;" in sent_text  # <script> should be escaped

    def test_fallback_from_markdownv2_to_html_on_error(self):
        """Test fallback mechanism from MarkdownV2 to HTML when formatting fails."""
        test_title = "Problematic Content"
        test_message = "This content causes MarkdownV2 parsing errors"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        # First call (MarkdownV2) fails, second call (HTML) succeeds
        mock_bot_instance.send_message = AsyncMock(
            side_effect=[Exception("MarkdownV2 parse error"), True]
        )

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should succeed after fallback
            assert result is True

            # Should have made two calls - first with MarkdownV2, then with HTML
            assert mock_bot_instance.send_message.call_count == 2

            # First call should be MarkdownV2
            first_call = mock_bot_instance.send_message.call_args_list[0]
            assert first_call[1]["parse_mode"] == "MarkdownV2"

            # Second call should be HTML fallback
            second_call = mock_bot_instance.send_message.call_args_list[1]
            assert second_call[1]["parse_mode"] == "HTML"

    def test_fallback_from_html_to_plain_text_on_error(self):
        """Test fallback mechanism from HTML to plain text when formatting fails."""
        test_title = "Problematic HTML Content"
        test_message = "This content causes HTML parsing errors"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="html",
        )

        mock_bot_instance = AsyncMock()
        # First call (HTML) fails, second call (plain text) succeeds
        mock_bot_instance.send_message = AsyncMock(
            side_effect=[Exception("HTML parse error"), True]
        )

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should succeed after fallback
            assert result is True

            # Should have made two calls - first with HTML, then with plain text
            assert mock_bot_instance.send_message.call_count == 2

            # First call should be HTML
            first_call = mock_bot_instance.send_message.call_args_list[0]
            assert first_call[1]["parse_mode"] == "HTML"

            # Second call should be plain text fallback
            second_call = mock_bot_instance.send_message.call_args_list[1]
            assert second_call[1]["parse_mode"] is None

    def test_complete_fallback_chain_markdownv2_to_plain_text(self):
        """Test complete fallback chain: MarkdownV2 -> HTML -> plain text."""
        test_title = "Extremely Problematic Content"
        test_message = "This content fails all formatting attempts"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        # First call (MarkdownV2) fails, second call (HTML) fails, third call (plain text) succeeds
        mock_bot_instance.send_message = AsyncMock(
            side_effect=[Exception("MarkdownV2 parse error"), Exception("HTML parse error"), True]
        )

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should succeed after complete fallback
            assert result is True

            # Should have made three calls
            assert mock_bot_instance.send_message.call_count == 3

            # Verify fallback sequence
            calls = mock_bot_instance.send_message.call_args_list
            assert calls[0][1]["parse_mode"] == "MarkdownV2"  # First attempt
            assert calls[1][1]["parse_mode"] == "HTML"  # First fallback
            assert calls[2][1]["parse_mode"] is None  # Final fallback (plain text)

    def test_fallback_maintains_synchronous_interface(self):
        """Test that fallback logic maintains synchronous interface throughout."""
        test_title = "Fallback Test"
        test_message = "Testing synchronous fallback behavior"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(
            side_effect=[Exception("First format failed"), True]
        )

        with patch("telegram.Bot", return_value=mock_bot_instance):
            # This should work synchronously despite multiple attempts
            result = config.send_message(title=test_title, message=test_message, logger=None)

            # Should return boolean result synchronously
            assert isinstance(result, bool)
            assert result is True

    def test_no_fallback_for_network_errors(self):
        """Test that network errors don't trigger format fallback."""
        test_title = "Network Error Test"
        test_message = "This should not trigger format fallback"

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        mock_bot_instance = AsyncMock()
        # Simulate network error (should not trigger fallback)
        mock_bot_instance.send_message = AsyncMock(side_effect=Exception("Network timeout"))

        with patch("telegram.Bot", return_value=mock_bot_instance):
            with patch("time.sleep") as mock_sleep:
                result = config.send_message(title=test_title, message=test_message, logger=None)

                # Should fail without fallback for network errors
                assert result is False

                # Should retry network errors 5 times (no fallback for network errors)
                assert mock_bot_instance.send_message.call_count == 5
                # Should sleep 4 times between retries
                assert mock_sleep.call_count == 4

                # All calls should use the same format (no fallback)
                for call in mock_bot_instance.send_message.call_args_list:
                    assert call[1]["parse_mode"] == "MarkdownV2"

    def test_plain_text_never_fails_formatting(self):
        """Test that plain text format never requires fallback."""
        test_title = "Plain Text with All Special Characters: *_[]()~`>#+-=|{}.!"
        test_message = "No escaping needed: <>&\"' and all other chars"

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

            # Should only make one call (no fallback needed)
            assert mock_bot_instance.send_message.call_count == 1

            call_args = mock_bot_instance.send_message.call_args
            assert call_args[1]["parse_mode"] is None

            # Text should be sent as-is without any escaping
            sent_text = call_args[1]["text"]
            assert "*_[]()~`>#+-=|{}.!" in sent_text
            assert "<>&\"'" in sent_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
