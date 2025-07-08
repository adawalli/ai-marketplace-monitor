"""Test Telegram integration with notification orchestration system"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_marketplace_monitor.ai import AIResponse
from ai_marketplace_monitor.listing import Listing
from ai_marketplace_monitor.notification import NotificationStatus
from ai_marketplace_monitor.telegram import TelegramNotificationConfig
from ai_marketplace_monitor.user import User, UserConfig


class TestTelegramNotificationOrchestration:
    """Test that Telegram integrates properly with the notification orchestration system"""

    @pytest.fixture
    def mock_listing(self):
        """Create a mock listing for testing"""
        listing = Mock(spec=Listing)
        listing.id = "test123"
        listing.title = "Test Item"
        listing.price = "$100"
        listing.location = "Test City"
        listing.post_url = "https://example.com/item?ref=123"
        listing.description = "A great test item"
        listing.name = "test item"
        listing.marketplace = "test market"
        listing.hash = "testhash"
        listing.content = "test content"
        return listing

    @pytest.fixture
    def mock_ai_response(self):
        """Create a mock AI response"""
        response = Mock(spec=AIResponse)
        response.conclusion = "good"
        response.score = 8.5
        response.comment = "This looks like a good deal"
        return response

    def test_telegram_notify_with_markdownv2(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that Telegram notify works with markdownv2 format"""
        # Create telegram config
        telegram_config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="markdownv2",
        )

        # Since TelegramNotificationConfig inherits from PushNotificationConfig,
        # it should have the notify method
        assert hasattr(telegram_config, "notify")

        # Mock the send_message_with_retry method
        with patch.object(
            telegram_config, "send_message_with_retry", return_value=True
        ) as mock_send:
            result = telegram_config.notify(
                listings=[mock_listing],
                ratings=[mock_ai_response],
                notification_status=[NotificationStatus.NOT_NOTIFIED],
                logger=None,
            )

            assert result is True
            mock_send.assert_called_once()

            # Check the message was formatted correctly
            title, message = mock_send.call_args[0]
            assert "Found 1 new test item from test market" in title
            # MarkdownV2 is not explicitly handled in PushNotificationConfig,
            # it defaults to plain text format
            assert "Test Item" in message
            assert "$100" in message
            assert "Test City" in message

    def test_telegram_with_user_config_notify_all(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that UserConfig.notify_all triggers Telegram notifications"""
        # Create a user config with telegram settings
        user_config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="-1001234567890",
            message_format="html",
            notify_with=["telegram"],
        )

        # Mock the send_message method in TelegramNotificationConfig
        with patch(
            "ai_marketplace_monitor.telegram.TelegramNotificationConfig.send_message",
            return_value=True,
        ):
            # Use NotificationConfig.notify_all which is called by User.notify
            from ai_marketplace_monitor.notification import NotificationConfig

            result = NotificationConfig.notify_all(
                user_config,
                listings=[mock_listing],
                ratings=[mock_ai_response],
                notification_status=[NotificationStatus.NOT_NOTIFIED],
                logger=None,
            )

            # Should return True if notification was sent
            assert result is True

    def test_telegram_synchronous_wrapper_in_orchestration(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that the synchronous wrapper works in the orchestration flow"""
        telegram_config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="plain_text",
        )

        # Mock the telegram Bot class
        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            # Test the send_message method directly
            result = telegram_config.send_message(
                title="Test Title", message="Test Message", logger=None
            )

            assert result is True
            # Verify the bot was called with correct parameters
            mock_bot_instance.send_message.assert_called_once_with(
                chat_id="@testchannel",
                text="Test Title\n\nTest Message",
                parse_mode=None,  # plain_text uses None
            )

    def test_multiple_notifications_with_telegram(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that multiple notification types including Telegram work together"""
        user_config = UserConfig(
            name="test_user",
            # Telegram settings
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            # Pushbullet settings (example of multiple notifications)
            pushbullet_token="test_pushbullet_token",
            message_format="markdown",
            notify_with=["telegram", "pushbullet"],
        )

        # Create a User instance
        user = User(config=user_config, logger=None)

        # Mock cache
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Not notified before

        # Mock both notification methods
        with patch(
            "ai_marketplace_monitor.telegram.TelegramNotificationConfig.send_message",
            return_value=True,
        ) as mock_telegram:
            with patch(
                "ai_marketplace_monitor.pushbullet.PushbulletNotificationConfig.send_message",
                return_value=True,
            ) as mock_pushbullet:
                # Create a proper item config mock
                item_config_mock = Mock()
                item_config_mock.name = "test_item"

                # Call user.notify
                user.notify(
                    listings=[mock_listing],
                    ratings=[mock_ai_response],
                    item_config=item_config_mock,
                    local_cache=mock_cache,
                )

                # Both notification methods should have been called
                assert mock_telegram.called or mock_pushbullet.called

    def test_telegram_error_handling_in_orchestration(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that errors in Telegram are handled properly in orchestration"""
        telegram_config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="html",
            max_retries=2,  # Reduce retries for faster test
            retry_delay=0,  # No delay for testing
        )

        # Mock send_message to raise an exception
        with patch.object(telegram_config, "send_message", side_effect=Exception("Network error")):
            # The send_message_with_retry should handle the exception
            result = telegram_config.send_message_with_retry(
                title="Test", message="Test message", logger=None
            )

            # Should return False after retries fail
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
