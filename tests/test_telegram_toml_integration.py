"""Test Telegram integration with TOML configuration parsing and UserConfig"""

import sys
import tempfile
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    pass
else:
    pass

from ai_marketplace_monitor.config import Config
from ai_marketplace_monitor.telegram import TelegramNotificationConfig


class TestTelegramTOMLIntegration:
    """Test TOML parsing and integration with UserConfig for Telegram notifications"""

    def test_telegram_config_in_toml(self) -> None:
        """Test that telegram configuration can be parsed from TOML"""
        toml_content = """
[marketplace.test_market]
city_name = ["Test City"]
search_city = ["test_city"]
radius = [10]
currency = ["USD"]

[user.test_user]
notify_with = ["telegram_notify"]

[item.test_item]
search_phrases = ["test search"]

[notification.telegram_notify]
telegram_bot_token = "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr"
telegram_chat_id = "@testchannel"
message_format = "markdownv2"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_file = Path(f.name)

        try:
            # Load config
            config = Config([config_file])

            # Check notification was parsed
            assert "telegram_notify" in config.notification
            telegram_config = config.notification["telegram_notify"]
            assert isinstance(telegram_config, TelegramNotificationConfig)
            assert (
                telegram_config.telegram_bot_token
                == "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr"
            )
            assert telegram_config.telegram_chat_id == "@testchannel"
            assert telegram_config.message_format == "markdownv2"

            # Check user config has telegram fields after expansion
            user_config = config.user["test_user"]
            assert (
                user_config.telegram_bot_token == "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr"
            )
            assert user_config.telegram_chat_id == "@testchannel"
            assert user_config.message_format == "markdownv2"

        finally:
            config_file.unlink()

    def test_multiple_notification_methods_with_telegram(self) -> None:
        """Test that multiple notification methods including telegram work together"""
        toml_content = """
[marketplace.test_market]
city_name = ["Test City"]
search_city = ["test_city"]
radius = [10]
currency = ["USD"]

[user.test_user]
notify_with = ["telegram_notify", "pushbullet_notify"]

[item.test_item]
search_phrases = ["test search"]

[notification.telegram_notify]
telegram_bot_token = "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr"
telegram_chat_id = "-1001234567890"
message_format = "html"

[notification.pushbullet_notify]
pushbullet_token = "test_pushbullet_token"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_file = Path(f.name)

        try:
            # Load config
            config = Config([config_file])

            # Check both notifications were parsed
            assert "telegram_notify" in config.notification
            assert "pushbullet_notify" in config.notification

            # Check user config has fields from both
            user_config = config.user["test_user"]
            assert (
                user_config.telegram_bot_token == "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr"
            )
            assert user_config.telegram_chat_id == "-1001234567890"
            assert user_config.pushbullet_token == "test_pushbullet_token"

        finally:
            config_file.unlink()

    def test_telegram_config_validation_in_toml(self) -> None:
        """Test that invalid telegram configuration in TOML triggers validation errors"""
        toml_content = """
[marketplace.test_market]
city_name = ["Test City"]
search_city = ["test_city"]
radius = [10]
currency = ["USD"]

[user.test_user]
notify_with = ["telegram_notify"]

[item.test_item]
search_phrases = ["test search"]

[notification.telegram_notify]
telegram_bot_token = "invalid_token_format"
telegram_chat_id = "@testchannel"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_file = Path(f.name)

        try:
            # Loading config should trigger validation error
            with pytest.raises(ValueError, match="Invalid telegram bot token format"):
                Config([config_file])
        finally:
            config_file.unlink()

    def test_telegram_disabled_notification(self) -> None:
        """Test that disabled telegram notifications are not expanded to users"""
        toml_content = """
[marketplace.test_market]
city_name = ["Test City"]
search_city = ["test_city"]
radius = [10]
currency = ["USD"]

[user.test_user]
notify_with = ["telegram_notify"]

[item.test_item]
search_phrases = ["test search"]

[notification.telegram_notify]
enabled = false
telegram_bot_token = "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqrr"
telegram_chat_id = "@testchannel"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_file = Path(f.name)

        try:
            # Load config
            config = Config([config_file])

            # Check notification was parsed but not expanded to user
            assert "telegram_notify" in config.notification
            user_config = config.user["test_user"]
            assert user_config.telegram_bot_token is None
            assert user_config.telegram_chat_id is None

        finally:
            config_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
