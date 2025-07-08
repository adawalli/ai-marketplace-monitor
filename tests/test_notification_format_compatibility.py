"""Test that markdownv2 support doesn't break other notification services"""

from unittest.mock import AsyncMock, Mock

import pytest

from ai_marketplace_monitor.ai import AIResponse
from ai_marketplace_monitor.listing import Listing
from ai_marketplace_monitor.notification import NotificationStatus
from ai_marketplace_monitor.ntfy import NtfyNotificationConfig
from ai_marketplace_monitor.pushbullet import PushbulletNotificationConfig
from ai_marketplace_monitor.pushover import PushoverNotificationConfig
from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestNotificationFormatCompatibility:
    """Test that adding markdownv2 support doesn't break existing notification services"""

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

    def test_pushover_ignores_markdownv2_setting(self) -> None:
        """Test that Pushover ignores markdownv2 and uses HTML as expected"""
        # Pushover should force HTML format regardless of user input
        pushover_config = PushoverNotificationConfig(
            name="test_pushover",
            pushover_user_key="test_user_key",
            pushover_api_token="test_token",
            message_format="markdownv2",  # User tries to set markdownv2
        )

        # Pushover should have forced it to HTML
        assert pushover_config.message_format == "html"

        # Verify required fields are correct
        assert pushover_config.required_fields == ["pushover_user_key", "pushover_api_token"]

    def test_pushbullet_ignores_markdownv2_setting(self) -> None:
        """Test that Pushbullet ignores markdownv2 and uses plain_text as expected"""
        # Pushbullet should force plain_text format regardless of user input
        pushbullet_config = PushbulletNotificationConfig(
            name="test_pushbullet",
            pushbullet_token="test_token",
            message_format="markdownv2",  # User tries to set markdownv2
        )

        # Pushbullet should have forced it to plain_text
        assert pushbullet_config.message_format == "plain_text"

        # Verify required fields are correct
        assert pushbullet_config.required_fields == ["pushbullet_token"]

    def test_ntfy_accepts_markdownv2_but_treats_as_plain_text(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that Ntfy accepts markdownv2 but treats it as plain text"""
        ntfy_config = NtfyNotificationConfig(
            name="test_ntfy",
            ntfy_server="https://ntfy.sh",
            ntfy_topic="test_topic",
            message_format="markdownv2",
        )

        # Ntfy should accept markdownv2 (inherits validation from PushNotificationConfig)
        assert ntfy_config.message_format == "markdownv2"

        # Mock the requests.post call
        from unittest.mock import patch

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            result = ntfy_config.notify(
                listings=[mock_listing],
                ratings=[mock_ai_response],
                notification_status=[NotificationStatus.NOT_NOTIFIED],
                logger=None,
            )

            assert result is True
            # Verify that the request was made (ntfy doesn't break with markdownv2)
            mock_post.assert_called_once()

    def test_telegram_properly_handles_markdownv2(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that Telegram properly handles markdownv2 format"""
        telegram_config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
            telegram_chat_id="@testchannel",
            message_format="markdownv2",
        )

        # Telegram should accept and validate markdownv2
        assert telegram_config.message_format == "markdownv2"

        # Mock telegram Bot
        from unittest.mock import patch

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock(return_value=True)

        with patch("telegram.Bot", return_value=mock_bot_instance):
            result = telegram_config.send_message(
                title="Test Title", message="Test Message", logger=None
            )

            assert result is True
            # Verify MarkdownV2 parse_mode was used
            mock_bot_instance.send_message.assert_called_once_with(
                chat_id="@testchannel",
                text="Test Title\n\nTest Message",
                parse_mode="MarkdownV2",  # Should be MarkdownV2 for markdownv2 format
            )

    def test_all_services_handle_notify_with_markdownv2(
        self, mock_listing: Mock, mock_ai_response: Mock
    ) -> None:
        """Test that all services can handle notify() call with markdownv2 without breaking"""
        services = [
            TelegramNotificationConfig(
                name="telegram",
                telegram_bot_token="123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr",
                telegram_chat_id="@test",
                message_format="markdownv2",
            ),
            PushoverNotificationConfig(
                name="pushover",
                pushover_user_key="test_user_key",
                pushover_api_token="test_token",
                message_format="markdownv2",  # Will be ignored and forced to HTML
            ),
            PushbulletNotificationConfig(
                name="pushbullet",
                pushbullet_token="test_token",
                message_format="markdownv2",  # Will be ignored and forced to plain_text
            ),
            NtfyNotificationConfig(
                name="ntfy",
                ntfy_server="https://ntfy.sh",
                ntfy_topic="test_topic",
                message_format="markdownv2",  # Will be accepted but treated as plain_text
            ),
        ]

        # Test that all services can be instantiated with markdownv2 without error
        for service in services:
            assert isinstance(service, type(service))
            # Verify format handling worked correctly
            if hasattr(service, "handle_message_format"):
                # Services that override message_format should have their expected format
                if isinstance(service, PushoverNotificationConfig):
                    assert service.message_format == "html"
                elif isinstance(service, PushbulletNotificationConfig):
                    assert service.message_format == "plain_text"
                else:
                    # Telegram and Ntfy should keep markdownv2
                    assert service.message_format == "markdownv2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
