"""Tests for Telegram message splitting functionality.

This module tests the message splitting logic that handles messages exceeding
Telegram's 4096 character limit by splitting them into multiple chunks while
maintaining formatting and synchronous interface.
"""

from unittest.mock import Mock, patch

import pytest

from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramMessageSplitting:
    """Test suite for message splitting logic."""

    def test_short_message_no_splitting(self):
        """Test that short messages are not split."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Short Title"
        message = "This is a short message that should not be split."

        # This should call the split_message method and return only one chunk
        chunks = config.split_message(title, message)

        assert len(chunks) == 1
        assert chunks[0] == f"{title}\n\n{message}"

    def test_long_message_splitting(self):
        """Test that long messages are split into multiple chunks."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Long Message Title"
        # Create a message longer than 4096 characters
        long_message = "This is a very long message. " * 150  # ~4500 characters

        chunks = config.split_message(title, long_message)

        assert len(chunks) > 1
        # Each chunk should be within the 4096 character limit
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_message_splitting_preserves_content(self):
        """Test that splitting preserves all content."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Test Title"
        message = "A" * 5000  # 5000 character message

        chunks = config.split_message(title, message)

        # Reconstruct the message from chunks
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                # First chunk contains title
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        assert reconstructed == message

    def test_send_message_with_long_content_calls_multiple_times(self):
        """Test that send_message calls the sending logic multiple times for long content."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Long Message Title"
        long_message = "This is a very long message. " * 150  # ~4500 characters

        # Mock the telegram Bot to track calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot

            # Mock asyncio.run to avoid actual async execution
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = True

                result = config.send_message(title, long_message)

                # Should succeed
                assert result is True
                # Should have been called multiple times (once for each chunk)
                assert mock_run.call_count > 1

    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # This should raise ValueError as per current implementation
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.split_message("Title", "")


class TestTelegramWordBoundarySplitting:
    """Test suite for word boundary splitting logic."""

    def test_word_boundary_splitting_basic(self):
        """Test that messages are split at word boundaries."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create a message that will split around word boundaries
        message = "word " * 1000  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1
        # Check that each chunk (except possibly the last) ends with a complete word
        for i, chunk in enumerate(chunks[:-1]):  # All except last chunk
            # Remove title from first chunk for content analysis
            if i == 0 and chunk.startswith(title):
                content = chunk[len(title + "\n\n") :]
            else:
                content = chunk

            # Should end with a space (complete word) or be at natural boundary
            assert content.endswith(" ") or content.endswith("\n")

    def test_word_boundary_splitting_with_punctuation(self):
        """Test word boundary splitting with punctuation."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create message with punctuation
        sentence = "This is a sentence with punctuation, including commas and periods. "
        message = sentence * 100  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1
        # Verify no words are split across chunks
        for i, chunk in enumerate(chunks[:-1]):
            if i == 0 and chunk.startswith(title):
                content = chunk[len(title + "\n\n") :]
            else:
                content = chunk

            # Should end at word boundary (space, punctuation, or newline)
            assert content[-1] in " .,!?;:\n"

    def test_word_boundary_splitting_preserves_words(self):
        """Test that individual words are never split."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create message with distinctive words
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        message = " ".join(words * 200)  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1

        # Reconstruct message and verify all words are intact
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        # All original words should be present and intact
        for word in words:
            assert word in reconstructed
            # Word should not be split (shouldn't find partial matches)
            import re

            matches = re.findall(r"\b" + re.escape(word) + r"\b", reconstructed)
            assert len(matches) == 200  # Should find exactly 200 instances

    def test_word_boundary_splitting_with_newlines(self):
        """Test word boundary splitting respects newlines."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create message with multiple paragraphs
        paragraph = "This is a paragraph with multiple sentences. It has several words.\n\n"
        message = paragraph * 100  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1

        # Verify that newlines are preserved and used as split points
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        # Should contain the original paragraph breaks
        assert "\n\n" in reconstructed


class TestTelegramFormatPreservation:
    """Test suite for format preservation across message chunks."""

    def test_markdownv2_format_preservation(self):
        """Test that MarkdownV2 formatting is preserved across chunks."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        title = "Title"
        # Create message with MarkdownV2 formatting - make it longer to force splitting
        formatted_text = "*Bold text* _italic text_ `code text` "
        message = formatted_text * 200  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1

        # Reconstruct and verify formatting is preserved
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        # Should contain the original formatting characters
        assert "*Bold text*" in reconstructed
        assert "_italic text_" in reconstructed
        assert "`code text`" in reconstructed

    def test_html_format_preservation(self):
        """Test that HTML formatting is preserved across chunks."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="html",
        )

        title = "Title"
        # Create message with HTML formatting
        formatted_text = "<b>Bold text</b> <i>italic text</i> <code>code text</code> "
        message = formatted_text * 100  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1

        # Reconstruct and verify formatting is preserved
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        # Should contain the original HTML tags
        assert "<b>Bold text</b>" in reconstructed
        assert "<i>italic text</i>" in reconstructed
        assert "<code>code text</code>" in reconstructed

    def test_plain_text_format_preservation(self):
        """Test that plain text formatting is preserved across chunks."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="plain_text",
        )

        title = "Title"
        # Create message with special characters that should be preserved
        formatted_text = "Text with special chars: @#$%^&*()_+-=[]{}|;':\",./<>? "
        message = formatted_text * 100  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        assert len(chunks) > 1

        # Reconstruct and verify all characters are preserved
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        # Should contain the original special characters
        assert "@#$%^&*()_+-=[]{}|;':\",./<>?" in reconstructed


class TestTelegramEdgeCases:
    """Test suite for edge cases including single long words."""

    def test_single_long_word_exceeding_limit(self):
        """Test handling of single words longer than 4096 characters."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create a single word longer than 4096 characters
        long_word = "a" * 5000
        message = f"Before {long_word} After"

        chunks = config.split_message(title, message)

        assert len(chunks) > 1

        # Reconstruct and verify the long word is handled appropriately
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            # Remove chunk indicator if present (format: "(1/2) ")
            import re

            chunk_without_indicator = re.sub(r"^\(\d+/\d+\) ", "", chunk)

            if i == 0 and chunk_without_indicator.startswith(title):
                reconstructed += chunk_without_indicator[len(title + "\n\n") :]
            else:
                reconstructed += chunk_without_indicator

        # The long word should be present (possibly split with continuation indicators)
        assert "Before" in reconstructed
        assert "After" in reconstructed
        assert "a" * 1000 in reconstructed  # Some portion of the long word

    def test_message_exactly_at_limit(self):
        """Test handling of messages exactly at the 4096 character limit."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create message that when combined with title is exactly 4096 chars
        title_with_separator = f"{title}\n\n"
        remaining_chars = 4096 - len(title_with_separator)
        message = "a" * remaining_chars

        chunks = config.split_message(title, message)

        # Should create exactly one chunk at the limit
        assert len(chunks) == 1
        assert len(chunks[0]) == 4096

    def test_empty_title_handling(self):
        """Test handling of empty titles."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Test with empty title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.split_message("", "Some message")

    def test_very_short_message_handling(self):
        """Test handling of very short messages."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        message = "Hi"

        chunks = config.split_message(title, message)

        # Should create exactly one chunk
        assert len(chunks) == 1
        assert chunks[0] == f"{title}\n\n{message}"

    def test_message_with_only_spaces(self):
        """Test handling of messages with only whitespace."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        message = "   "  # Only spaces

        # Should raise ValueError for empty message after stripping
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.split_message(title, message)

    def test_chunk_indicators_presence(self):
        """Test that chunk indicators are properly added to multi-chunk messages."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        title = "Title"
        # Create a message that will require multiple chunks
        message = "This is a long message. " * 200  # Should be longer than 4096 characters

        chunks = config.split_message(title, message)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check that each chunk has the proper indicator
        import re

        for i, chunk in enumerate(chunks):
            expected_indicator = f"({i+1}/{len(chunks)}) "
            assert chunk.startswith(
                expected_indicator
            ), f"Chunk {i+1} missing indicator: {chunk[:20]}..."

            # Verify the indicator format is correct
            indicator_match = re.match(r"^\(\d+/\d+\) ", chunk)
            assert (
                indicator_match is not None
            ), f"Invalid indicator format in chunk {i+1}: {chunk[:20]}..."
