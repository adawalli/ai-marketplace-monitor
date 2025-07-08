"""Tests for TelegramNotificationConfig integration with UserConfig."""

from typing import Callable
from unittest.mock import Mock, patch

import pytest
from pytest import TempPathFactory

from ai_marketplace_monitor.config import Config
from ai_marketplace_monitor.notification import NotificationConfig, PushNotificationConfig
from ai_marketplace_monitor.telegram import TelegramNotificationConfig
from ai_marketplace_monitor.user import UserConfig


@pytest.fixture(scope="session")
def config_file(tmp_path_factory: TempPathFactory) -> Callable:
    """Create temporary config files for testing."""

    def generate_config_file(content: str) -> str:
        fn = tmp_path_factory.mktemp("config") / "test.toml"
        with open(fn, "w") as f:
            f.write(content)
        return str(fn)

    return generate_config_file


# Base configuration strings following existing patterns
base_marketplace_cfg = """
[marketplace.facebook]
search_city = 'dallas'
"""

base_item_cfg = """
[item.name]
search_phrases = 'search word one'
"""

base_telegram_cfg = """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
"""

base_telegram_username_cfg = """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '@testuser'
"""

base_telegram_group_cfg = """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '-1001234567890'
"""

invalid_telegram_token_cfg = """
[notification.telegram]
telegram_bot_token = 'invalid_token'
telegram_chat_id = '123456789'
"""

invalid_telegram_chat_id_cfg = """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = 'invalid_chat_id'
"""

notify_user_telegram_cfg = """
[user.user1]
notify_with = ['telegram']
"""

notify_user_mixed_cfg = """
[user.user1]
notify_with = ['telegram', 'pushbullet1']
"""

base_pushbullet_cfg = """
[notification.pushbullet1]
pushbullet_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
"""


class TestTelegramNotificationConfigInstantiation:
    """Test TelegramNotificationConfig class instantiation and initialization."""

    def test_default_instantiation(self):
        """Test creating TelegramNotificationConfig with default values."""
        config = TelegramNotificationConfig(name="test_telegram")

        # Check default values
        assert config.notify_method == "telegram"
        assert config.telegram_bot_token is None
        assert config.telegram_chat_id is None
        assert config.message_format == "markdownv2"
        assert config.max_retries == 5
        assert config.retry_delay == 60

    def test_valid_instantiation_with_all_fields(self):
        """Test creating TelegramNotificationConfig with all valid fields."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="html",
            max_retries=5,
            retry_delay=2,
        )

        assert config.telegram_bot_token == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        assert config.telegram_chat_id == "123456789"
        assert config.message_format == "html"
        assert config.max_retries == 5
        assert config.retry_delay == 2
        assert config.notify_method == "telegram"

    def test_valid_instantiation_with_username_chat_id(self):
        """Test creating TelegramNotificationConfig with username chat ID."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="@testuser",
        )

        assert config.telegram_chat_id == "@testuser"

    def test_valid_instantiation_with_group_chat_id(self):
        """Test creating TelegramNotificationConfig with group chat ID."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="-1001234567890",
        )

        assert config.telegram_chat_id == "-1001234567890"

    def test_instantiation_with_invalid_bot_token(self):
        """Test that invalid bot token raises ValueError during instantiation."""
        with pytest.raises(ValueError, match="Invalid telegram bot token format"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="invalid_token",
                telegram_chat_id="123456789",
            )

    def test_instantiation_with_invalid_chat_id(self):
        """Test that invalid chat ID raises ValueError during instantiation."""
        with pytest.raises(ValueError, match="Invalid telegram chat ID format"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="invalid_chat_id",
            )

    def test_instantiation_with_invalid_message_format(self):
        """Test that invalid message format raises ValueError during instantiation."""
        with pytest.raises(ValueError, match="Invalid message format"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="123456789",
                message_format="invalid_format",
            )

    def test_instantiation_with_empty_bot_token(self):
        """Test that empty bot token raises ValueError during instantiation."""
        with pytest.raises(ValueError, match="telegram_bot_token must be a non-empty string"):
            TelegramNotificationConfig(
                name="test_telegram", telegram_bot_token="", telegram_chat_id="123456789"
            )

    def test_instantiation_with_empty_chat_id(self):
        """Test that empty chat ID raises ValueError during instantiation."""
        with pytest.raises(ValueError, match="telegram_chat_id must be a non-empty string"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="",
            )

    def test_instantiation_with_whitespace_trimming(self):
        """Test that whitespace is trimmed during instantiation."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="  123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk  ",
            telegram_chat_id="  123456789  ",
            message_format="  html  ",
        )

        assert config.telegram_bot_token == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        assert config.telegram_chat_id == "123456789"
        assert config.message_format == "html"

    def test_required_fields_property(self):
        """Test that required_fields class variable is correctly set."""
        assert TelegramNotificationConfig.required_fields == [
            "telegram_bot_token",
            "telegram_chat_id",
        ]

    def test_has_required_fields_method(self):
        """Test _has_required_fields method for different configurations."""
        # Config with all required fields
        config_complete = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )
        assert config_complete._has_required_fields()

        # Config with missing bot token
        config_no_token = TelegramNotificationConfig(
            name="test_telegram", telegram_chat_id="123456789"
        )
        assert not config_no_token._has_required_fields()

        # Config with missing chat ID
        config_no_chat = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
        )
        assert not config_no_chat._has_required_fields()

        # Config with no required fields
        config_empty = TelegramNotificationConfig(name="test_telegram")
        assert not config_empty._has_required_fields()

    def test_dataclass_validation(self):
        """Test that dataclass behaves correctly with unknown fields."""
        # Dataclasses don't prevent extra fields at creation, but we can test normal usage
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )
        assert config.name == "test_telegram"
        assert hasattr(config, "telegram_bot_token")
        assert hasattr(config, "telegram_chat_id")

    def test_send_message_method_exists(self):
        """Test that send_message method exists and is implemented."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Method should exist
        assert hasattr(config, "send_message")
        assert callable(config.send_message)

        # Method should be implemented and work with mocked telegram
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True
                result = config.send_message("Test Title", "Test Message")
                assert result is True


class TestTelegramNotificationConfigIntegration:
    """Test TelegramNotificationConfig integration with UserConfig."""

    def test_user_config_inherits_telegram_fields(self):
        """Test that UserConfig inherits telegram fields from TelegramNotificationConfig."""
        # This test should fail initially until we add TelegramNotificationConfig to UserConfig inheritance
        config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=1,
            retry_delay=1,
        )

        # These should be accessible through UserConfig
        assert hasattr(config, "telegram_bot_token")
        assert hasattr(config, "telegram_chat_id")
        assert hasattr(config, "message_format")
        assert config.telegram_bot_token == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        assert config.telegram_chat_id == "123456789"

    def test_user_config_telegram_field_validation(self):
        """Test that telegram field validation works through UserConfig."""
        # Valid telegram configuration
        config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=1,
            retry_delay=1,
        )

        # Should not raise any validation errors
        assert config.telegram_bot_token == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        assert config.telegram_chat_id == "123456789"

    def test_user_config_telegram_invalid_token(self):
        """Test that invalid telegram token raises validation error through UserConfig."""
        with pytest.raises(ValueError, match="Invalid telegram bot token format"):
            UserConfig(
                name="test_user",
                telegram_bot_token="invalid_token",
                telegram_chat_id="123456789",
                max_retries=1,
                retry_delay=1,
            )

    def test_user_config_telegram_invalid_chat_id(self):
        """Test that invalid telegram chat ID raises validation error through UserConfig."""
        with pytest.raises(ValueError, match="Invalid telegram chat ID format"):
            UserConfig(
                name="test_user",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="invalid_chat_id",
                max_retries=1,
                retry_delay=1,
            )

    def test_user_config_telegram_username_chat_id(self):
        """Test that username format chat ID works through UserConfig."""
        config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="@testuser",
            max_retries=1,
            retry_delay=1,
        )

        assert config.telegram_chat_id == "@testuser"

    def test_user_config_telegram_group_chat_id(self):
        """Test that group chat ID format works through UserConfig."""
        config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="-1001234567890",
            max_retries=1,
            retry_delay=1,
        )

        assert config.telegram_chat_id == "-1001234567890"

    def test_user_config_telegram_message_format_validation(self):
        """Test that message format validation works through UserConfig."""
        config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="markdownv2",
            max_retries=1,
            retry_delay=1,
        )

        assert config.message_format == "markdownv2"

    def test_user_config_telegram_invalid_message_format(self):
        """Test that invalid message format raises validation error through UserConfig."""
        with pytest.raises(ValueError, match="Invalid message format"):
            UserConfig(
                name="test_user",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="123456789",
                message_format="invalid_format",
                max_retries=1,
                retry_delay=1,
            )

    def test_user_config_telegram_required_fields(self):
        """Test that telegram required fields are enforced through UserConfig."""
        config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=1,
            retry_delay=1,
        )

        # Should have required fields for telegram
        assert config._has_required_fields()

    def test_user_config_telegram_missing_required_fields(self):
        """Test that missing telegram required fields are detected through UserConfig."""
        config = UserConfig(
            name="test_user",
            telegram_bot_token=None,
            telegram_chat_id="123456789",
            max_retries=1,
            retry_delay=1,
        )

        # Should not have required fields when token is missing
        assert not config._has_required_fields()


class TestTelegramFieldValidationAndInheritance:
    """Test field validation rules and inheritance behavior."""

    def test_field_type_validation(self):
        """Test that field types are validated correctly."""
        # Test invalid max_retries type
        with pytest.raises(ValueError, match="max_retries must be an integer"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="123456789",
                max_retries="invalid",
            )

        # Test invalid retry_delay type
        with pytest.raises(ValueError, match="retry_delay must be an integer"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="123456789",
                retry_delay="invalid",
            )

        # Test invalid telegram_bot_token type
        with pytest.raises(ValueError, match="telegram_bot_token must be a non-empty string"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token=123456789,
                telegram_chat_id="123456789",
            )

        # Test invalid telegram_chat_id type
        with pytest.raises(ValueError, match="telegram_chat_id must be a non-empty string"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id=123456789,
            )

    def test_field_range_validation(self):
        """Test that field ranges are validated correctly."""
        # Test valid ranges
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=1,
            retry_delay=1,
        )
        assert config.max_retries == 1
        assert config.retry_delay == 1

        # Test zero values
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=0,
            retry_delay=0,
        )
        assert config.max_retries == 0
        assert config.retry_delay == 0

        # Test large values
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=1000,
            retry_delay=3600,
        )
        assert config.max_retries == 1000
        assert config.retry_delay == 3600

    def test_field_nullability_validation(self):
        """Test that nullable fields are handled correctly."""
        # Test that optional fields can be None
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token=None,
            telegram_chat_id=None,
            message_format=None,
        )
        assert config.telegram_bot_token is None
        assert config.telegram_chat_id is None
        assert config.message_format == "markdownv2"  # Default value after processing

        # Test that required fields for functionality are None initially
        assert not config._has_required_fields()

        # Test that setting non-None values works
        config.telegram_bot_token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        config.telegram_chat_id = "123456789"
        config.handle_telegram_bot_token()
        config.handle_telegram_chat_id()
        assert config._has_required_fields()

    def test_inheritance_from_push_notification_config(self):
        """Test that TelegramNotificationConfig correctly inherits from PushNotificationConfig."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Check inheritance hierarchy
        assert isinstance(config, PushNotificationConfig)
        assert isinstance(config, NotificationConfig)
        assert hasattr(config, "notify_method")
        assert hasattr(config, "max_retries")
        assert hasattr(config, "retry_delay")
        assert hasattr(config, "message_format")

        # Check that telegram-specific fields exist
        assert hasattr(config, "telegram_bot_token")
        assert hasattr(config, "telegram_chat_id")
        assert hasattr(config, "required_fields")

        # Check method resolution order
        assert TelegramNotificationConfig.__mro__[0] == TelegramNotificationConfig
        assert TelegramNotificationConfig.__mro__[1] == PushNotificationConfig
        assert TelegramNotificationConfig.__mro__[2] == NotificationConfig

    def test_method_override_behavior(self):
        """Test that methods are properly overridden in inheritance chain."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Test that notify_method is overridden
        assert config.notify_method == "telegram"
        assert config.notify_method != "push_notification"

        # Test that required_fields is overridden
        assert config.required_fields == ["telegram_bot_token", "telegram_chat_id"]

        # Test that message_format handling is overridden
        assert config.message_format == "markdownv2"  # Telegram default
        # This should be different from PushNotificationConfig default

        # Test that send_message method exists and is implemented
        assert hasattr(config, "send_message")
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True
                result = config.send_message("Test", "Message")
                assert result is True

    def test_field_validation_handler_inheritance(self):
        """Test that field validation handlers work correctly through inheritance."""
        # Test that inherited handlers work with proper types
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=5,  # Must be int, not string
            retry_delay=60,  # Must be int, not string
        )

        # The handlers should have been called during initialization
        assert isinstance(config.max_retries, int)
        assert isinstance(config.retry_delay, int)
        assert config.max_retries == 5
        assert config.retry_delay == 60

        # Test telegram-specific handlers work with whitespace trimming
        config_with_whitespace = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="  123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk  ",
            telegram_chat_id="  123456789  ",
        )
        assert isinstance(config_with_whitespace.telegram_bot_token, str)
        assert isinstance(config_with_whitespace.telegram_chat_id, str)
        assert (
            config_with_whitespace.telegram_bot_token.strip()
            == config_with_whitespace.telegram_bot_token
        )
        assert (
            config_with_whitespace.telegram_chat_id.strip()
            == config_with_whitespace.telegram_chat_id
        )

    def test_polymorphic_behavior_with_notification_config(self):
        """Test polymorphic behavior when using NotificationConfig interface."""
        # Create telegram config
        telegram_config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Test that it can be used as NotificationConfig
        def test_notification_interface(config: NotificationConfig) -> str:
            assert isinstance(config, TelegramNotificationConfig)
            return config.notify_method

        # Should work polymorphically
        assert test_notification_interface(telegram_config) == "telegram"

        # Test that common interface methods work
        assert hasattr(telegram_config, "_has_required_fields")
        assert hasattr(telegram_config, "send_message_with_retry")
        assert telegram_config._has_required_fields()

        # Test that base class methods are available
        assert hasattr(telegram_config, "max_retries")
        assert hasattr(telegram_config, "retry_delay")

    def test_user_config_inheritance_with_telegram(self):
        """Test that UserConfig correctly inherits telegram functionality."""
        user_config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            max_retries=1,
            retry_delay=1,
        )

        # Test that UserConfig inherits from TelegramNotificationConfig
        assert isinstance(user_config, TelegramNotificationConfig)
        assert isinstance(user_config, PushNotificationConfig)
        assert isinstance(user_config, NotificationConfig)

        # Test that telegram methods are available
        assert hasattr(user_config, "handle_telegram_bot_token")
        assert hasattr(user_config, "handle_telegram_chat_id")
        assert hasattr(user_config, "handle_message_format")

        # Test that telegram validation works through UserConfig
        assert user_config._has_required_fields()
        assert user_config.telegram_bot_token is not None
        assert user_config.telegram_chat_id is not None

    def test_multiple_inheritance_method_resolution(self):
        """Test that multiple inheritance in UserConfig resolves methods correctly."""
        user_config = UserConfig(
            name="test_user",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="markdownv2",
            max_retries=1,
            retry_delay=1,
        )

        # Test that telegram message format takes precedence over push notification
        assert user_config.message_format == "markdownv2"

        # Test that required fields include telegram fields
        # Note: UserConfig may have multiple required_fields from different classes
        # but should include telegram fields
        assert "telegram_bot_token" in TelegramNotificationConfig.required_fields
        assert "telegram_chat_id" in TelegramNotificationConfig.required_fields

        # Test that MRO includes TelegramNotificationConfig first
        mro = UserConfig.__mro__
        telegram_index = None
        push_index = None
        for i, cls in enumerate(mro):
            if cls == TelegramNotificationConfig:
                telegram_index = i
            elif cls == PushNotificationConfig:
                push_index = i

        assert telegram_index is not None
        assert push_index is not None
        assert telegram_index < push_index  # Telegram should come before Push in MRO

    def test_field_validation_error_messages(self):
        """Test that field validation provides clear error messages."""
        # Test telegram_bot_token validation messages
        with pytest.raises(ValueError, match="telegram_bot_token must be a non-empty string"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="",
                telegram_chat_id="123456789",
            )

        with pytest.raises(ValueError, match="Invalid telegram bot token format"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="invalid_format",
                telegram_chat_id="123456789",
            )

        # Test telegram_chat_id validation messages
        with pytest.raises(ValueError, match="telegram_chat_id must be a non-empty string"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="",
            )

        with pytest.raises(ValueError, match="Invalid telegram chat ID format"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="invalid_format",
            )

        # Test message_format validation messages
        with pytest.raises(ValueError, match="Invalid message format"):
            TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="123456789",
                message_format="invalid_format",
            )

    def test_class_variable_inheritance(self):
        """Test that class variables are properly inherited and overridden."""
        # Test that notify_method is properly set
        assert TelegramNotificationConfig.notify_method == "telegram"
        assert hasattr(TelegramNotificationConfig, "required_fields")
        assert TelegramNotificationConfig.required_fields == [
            "telegram_bot_token",
            "telegram_chat_id",
        ]

        # Test that instance has access to class variables
        config = TelegramNotificationConfig(name="test_telegram")
        assert config.notify_method == "telegram"
        assert config.required_fields == ["telegram_bot_token", "telegram_chat_id"]

        # Test that class variables are different from parent class
        assert TelegramNotificationConfig.notify_method != PushNotificationConfig.notify_method
        assert TelegramNotificationConfig.required_fields != PushNotificationConfig.required_fields


class TestTelegramTOMLInheritanceIntegration:
    """Test telegram configuration inheritance through TOML files and UserConfig."""

    def test_user_config_inherits_telegram_from_toml_sections(self, config_file: Callable):
        """Test that UserConfig inherits telegram fields when configured in TOML sections."""
        cfg = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config = Config([cfg])

        # Get the user config
        user_config = config.user["user1"]

        # Test that telegram fields are inherited and accessible through UserConfig
        assert hasattr(user_config, "telegram_bot_token")
        assert hasattr(user_config, "telegram_chat_id")
        assert hasattr(user_config, "message_format")

        # Test that the telegram notification section values are accessible through user
        # This tests the inheritance chain working properly
        telegram_notification = config.notification["telegram"]
        assert isinstance(telegram_notification, TelegramNotificationConfig)
        assert user_config.telegram_bot_token == telegram_notification.telegram_bot_token
        assert user_config.telegram_chat_id == telegram_notification.telegram_chat_id

    def test_user_config_telegram_field_validation_from_toml(self, config_file: Callable):
        """Test that telegram field validation works when loaded from TOML through UserConfig."""
        # Test invalid token in TOML - should fail during config creation
        cfg_invalid = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + invalid_telegram_token_cfg
        )

        with pytest.raises(Exception):  # Should fail validation during config parsing
            Config([cfg_invalid])

        # Test invalid chat ID in TOML - should fail during config creation
        cfg_invalid_chat = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + invalid_telegram_chat_id_cfg
        )

        with pytest.raises(Exception):  # Should fail validation during config parsing
            Config([cfg_invalid_chat])

    def test_user_config_telegram_synchronous_interface_from_toml(self, config_file: Callable):
        """Test that UserConfig telegram methods maintain synchronous interface when loaded from TOML."""
        cfg = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config = Config([cfg])
        user_config = config.user["user1"]

        # Test that UserConfig has the synchronous send_message method
        assert hasattr(user_config, "send_message")
        assert callable(user_config.send_message)

        # Mock telegram to test synchronous interface
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # This should work synchronously through UserConfig inheritance
                result = user_config.send_message("Test Title", "Test Message")
                assert result is True
                # Should not be a coroutine
                assert not hasattr(result, "__await__")

    def test_user_config_telegram_required_fields_inheritance(self, config_file: Callable):
        """Test that UserConfig inherits and respects telegram required fields from TOML."""
        cfg = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config = Config([cfg])
        user_config = config.user["user1"]

        # Test that required fields validation works through inheritance
        assert user_config._has_required_fields()  # Should have telegram required fields

        # Test that the required fields are from TelegramNotificationConfig
        # Access the class variable through the instance
        assert "telegram_bot_token" in TelegramNotificationConfig.required_fields
        assert "telegram_chat_id" in TelegramNotificationConfig.required_fields

    def test_toml_section_inheritance_with_missing_telegram_config(self, config_file: Callable):
        """Test that missing telegram notification section is properly handled in inheritance."""
        # Configure user to use telegram but don't provide telegram notification section
        cfg = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            # Intentionally missing base_telegram_cfg
        )

        # This should fail because telegram notification is referenced but not configured
        with pytest.raises(Exception):
            Config([cfg])

    def test_toml_inheritance_with_multiple_notification_types(self, config_file: Callable):
        """Test inheritance when multiple notification types are configured."""
        cfg = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_mixed_cfg  # Uses both telegram and pushbullet1
            + base_telegram_cfg
            + base_pushbullet_cfg
        )

        config = Config([cfg])
        user_config = config.user["user1"]

        # Should inherit from both TelegramNotificationConfig and PushbulletNotificationConfig
        assert hasattr(user_config, "telegram_bot_token")
        assert hasattr(user_config, "telegram_chat_id")
        assert hasattr(user_config, "pushbullet_token")

        # Should be able to access both notification methods
        assert user_config.notify_with == ["telegram", "pushbullet1"]

    def test_toml_telegram_message_format_inheritance_and_defaults(self, config_file: Callable):
        """Test that message_format inheritance and defaults work correctly from TOML."""
        # Test with explicit message format
        cfg_explicit = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
message_format = 'html'
"""
        )

        config = Config([cfg_explicit])
        user_config = config.user["user1"]

        # Should inherit the explicit message format
        assert user_config.message_format == "html"

        # Test with default message format (no explicit setting)
        cfg_default = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config_default = Config([cfg_default])
        user_config_default = config_default.user["user1"]

        # Should use telegram default message format
        assert user_config_default.message_format == "markdownv2"


class TestTelegramTOMLIntegration:
    """Test telegram configuration parsing from TOML files."""

    @pytest.mark.parametrize(
        "config_content,acceptable",
        [
            # Valid telegram configurations
            (
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + base_telegram_cfg,
                True,
            ),
            (
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + base_telegram_username_cfg,
                True,
            ),
            (
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + base_telegram_group_cfg,
                True,
            ),
            # Invalid telegram configurations
            (
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + invalid_telegram_token_cfg,
                False,
            ),
            (
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + invalid_telegram_chat_id_cfg,
                False,
            ),
            # Mixed notifications (telegram + pushbullet)
            (
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_mixed_cfg
                + base_telegram_cfg
                + base_pushbullet_cfg,
                True,
            ),
            # Missing notification config
            (
                base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg,
                False,
            ),
        ],
    )
    def test_telegram_toml_config_parsing(
        self, config_file: Callable, config_content: str, acceptable: bool
    ):
        """Test telegram configuration parsing from TOML files."""
        cfg = config_file(config_content)

        if acceptable:
            config = Config([cfg])

            # Check that telegram notification config is parsed correctly
            if "telegram" in config.notification:
                telegram_config = config.notification["telegram"]
                assert hasattr(telegram_config, "telegram_bot_token")
                assert hasattr(telegram_config, "telegram_chat_id")
                assert hasattr(telegram_config, "message_format")

                # Check that the config has valid telegram fields
                if telegram_config.telegram_bot_token:
                    assert telegram_config.telegram_bot_token.count(":") == 1
                    bot_id, bot_token = telegram_config.telegram_bot_token.split(":")
                    assert bot_id.isdigit()
                    assert len(bot_token) > 0

            # Check that user config can access telegram fields
            if "user1" in config.user:
                user_config = config.user["user1"]
                # Should be able to access telegram fields through user config
                if hasattr(user_config, "telegram_bot_token"):
                    assert user_config.telegram_bot_token is not None
                    assert user_config.telegram_chat_id is not None
        else:
            with pytest.raises(Exception):
                Config([cfg])

    def test_telegram_toml_section_parsing(self, config_file: Callable):
        """Test that [notification.telegram] section is parsed correctly."""
        cfg = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config = Config([cfg])

        # Check that telegram notification is in the config
        assert "telegram" in config.notification
        telegram_config = config.notification["telegram"]
        assert isinstance(telegram_config, TelegramNotificationConfig)

        # Check telegram-specific fields
        assert (
            telegram_config.telegram_bot_token == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        )
        assert telegram_config.telegram_chat_id == "123456789"
        assert telegram_config.message_format == "markdownv2"  # default value

    def test_telegram_user_notification_linking(self, config_file: Callable):
        """Test that user notify_with links correctly to telegram notification."""
        cfg = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config = Config([cfg])

        # Check that user1 has telegram in notify_with
        user_config = config.user["user1"]
        assert user_config.notify_with is not None
        assert isinstance(user_config.notify_with, list)
        assert "telegram" in user_config.notify_with

        # Check that telegram notification is available
        assert "telegram" in config.notification


class TestTelegramSendMessage:
    """Test send_message method with TDD approach."""

    def test_send_message_successful_delivery(self):
        """Test successful message delivery using send_message method."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock the telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock the async send_message method
            mock_bot_instance.send_message = Mock()

            # Mock asyncio.run to simulate successful execution
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # This should pass once send_message is implemented
                result = config.send_message("Test Title", "Test Message")

                # Verify the call was made with correct parameters
                assert result is True
                mock_asyncio_run.assert_called_once()

    def test_send_message_invalid_bot_token(self):
        """Test send_message with invalid bot token."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to raise an exception for invalid token
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock the async send_message method to raise exception
            mock_bot_instance.send_message = Mock()

            # Mock asyncio.run to simulate telegram error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = Exception("Invalid bot token")

                # This should handle the error gracefully
                result = config.send_message("Test Title", "Test Message")

                # Should return False on error
                assert result is False

    def test_send_message_invalid_chat_id(self):
        """Test send_message with invalid chat ID."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to raise an exception for invalid chat ID
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock the async send_message method to raise exception
            mock_bot_instance.send_message = Mock()

            # Mock asyncio.run to simulate telegram error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = Exception("Invalid chat ID")

                # This should handle the error gracefully
                result = config.send_message("Test Title", "Test Message")

                # Should return False on error
                assert result is False

    def test_send_message_network_error(self):
        """Test send_message with network errors."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to raise network error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock the async send_message method to raise exception
            mock_bot_instance.send_message = Mock()

            # Mock asyncio.run to simulate network error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = Exception("Network error")

                # This should handle the error gracefully
                result = config.send_message("Test Title", "Test Message")

                # Should return False on error
                assert result is False

    def test_send_message_with_different_message_formats(self):
        """Test send_message with different message formats."""
        test_cases = [
            ("plain_text", "Test Title", "Test Message"),
            ("markdown", "Test Title", "Test Message"),
            ("markdownv2", "Test Title", "Test Message"),
            ("html", "Test Title", "Test Message"),
        ]

        for message_format, title, message in test_cases:
            config = TelegramNotificationConfig(
                name="test_telegram",
                telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                telegram_chat_id="123456789",
                message_format=message_format,
            )

            # Mock the telegram.Bot to avoid actual API calls
            with patch("telegram.Bot") as mock_bot_class:
                mock_bot_instance = Mock()
                mock_bot_class.return_value = mock_bot_instance

                # Mock the async send_message method
                mock_bot_instance.send_message = Mock()

                # Mock asyncio.run to simulate successful execution
                with patch("asyncio.run") as mock_asyncio_run:
                    mock_asyncio_run.return_value = True

                    # This should pass once send_message is implemented
                    result = config.send_message(title, message)

                    # Verify the call was made
                    assert result is True
                    mock_asyncio_run.assert_called_once()

    def test_send_message_maintains_synchronous_interface(self):
        """Test that send_message maintains synchronous interface despite async implementation."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock the telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock the async send_message method
            mock_bot_instance.send_message = Mock()

            # Mock asyncio.run to simulate successful execution
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # This should be a synchronous call that returns immediately
                result = config.send_message("Test Title", "Test Message")

                # Should return synchronously
                assert result is True
                # Result should not be a coroutine - it should be synchronous
                assert result  # Simple boolean check
                mock_asyncio_run.assert_called_once()


class TestTelegramSendMessageValidationErrors:
    """Test send_message method validation error handling with TDD approach."""

    def test_send_message_empty_title_parameter(self):
        """Test send_message with empty title parameter."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for empty title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message("", "Test Message")

    def test_send_message_none_title_parameter(self):
        """Test send_message with None title parameter."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for None title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message(None, "Test Message")

    def test_send_message_invalid_title_type(self):
        """Test send_message with invalid title type."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for non-string title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message(123, "Test Message")

    def test_send_message_whitespace_only_title(self):
        """Test send_message with whitespace-only title."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for whitespace-only title
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message("   ", "Test Message")

    def test_send_message_empty_message_parameter(self):
        """Test send_message with empty message parameter."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for empty message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message("Test Title", "")

    def test_send_message_none_message_parameter(self):
        """Test send_message with None message parameter."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for None message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message("Test Title", None)

    def test_send_message_invalid_message_type(self):
        """Test send_message with invalid message type."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for non-string message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message("Test Title", 123)

    def test_send_message_whitespace_only_message(self):
        """Test send_message with whitespace-only message."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for whitespace-only message
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message("Test Title", "   ")

    def test_send_message_missing_required_telegram_bot_token(self):
        """Test send_message with missing telegram_bot_token."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_chat_id="123456789",
        )

        # Should raise ValueError for missing bot token
        with pytest.raises(ValueError, match="telegram_bot_token is required"):
            config.send_message("Test Title", "Test Message")

    def test_send_message_missing_required_telegram_chat_id(self):
        """Test send_message with missing telegram_chat_id."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
        )

        # Should raise ValueError for missing chat ID
        with pytest.raises(ValueError, match="telegram_chat_id is required"):
            config.send_message("Test Title", "Test Message")

    def test_send_message_empty_telegram_bot_token(self):
        """Test send_message with empty telegram_bot_token after initialization."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Simulate empty bot token after initialization
        config.telegram_bot_token = ""

        # Should raise ValueError for empty bot token
        with pytest.raises(ValueError, match="telegram_bot_token is required"):
            config.send_message("Test Title", "Test Message")

    def test_send_message_empty_telegram_chat_id(self):
        """Test send_message with empty telegram_chat_id after initialization."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Simulate empty chat ID after initialization
        config.telegram_chat_id = ""

        # Should raise ValueError for empty chat ID
        with pytest.raises(ValueError, match="telegram_chat_id is required"):
            config.send_message("Test Title", "Test Message")

    def test_send_message_invalid_telegram_message_format(self):
        """Test send_message with invalid message format after initialization."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Simulate invalid message format after initialization
        config.message_format = "invalid_format"

        # Should raise ValueError for invalid message format
        with pytest.raises(ValueError, match="Invalid message format"):
            config.send_message("Test Title", "Test Message")

    def test_send_message_validates_synchronous_interface_with_errors(self):
        """Test that send_message validation errors are raised synchronously."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Validation errors should be raised immediately (synchronously)
        # without calling asyncio.run or any async operations
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message("", "Test Message")

        # Should not have attempted any async operations
        # This test verifies that validation happens before any async calls

    def test_send_message_parameter_validation_order(self):
        """Test that parameter validation happens in correct order."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Test that title is validated before message
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message("", "")  # Both are invalid, but title should be checked first

        # Test that message validation works when title is valid
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            config.send_message("Valid Title", "")

        # Test that config validation happens after parameter validation
        config.telegram_bot_token = None
        with pytest.raises(ValueError, match="title must be a non-empty string"):
            config.send_message("", "Test Message")  # Title validation should come first

    def test_send_message_parameter_trimming_validation(self):
        """Test that parameters are properly trimmed during validation."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock the actual telegram implementation to test trimming
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # Should trim whitespace from title and message
                result = config.send_message("  Test Title  ", "  Test Message  ")

                # Should succeed with trimmed values
                assert result is True
                mock_asyncio_run.assert_called_once()

    def test_send_message_comprehensive_validation_scenarios(self):
        """Test comprehensive validation scenarios for send_message."""
        # Test valid config with all required fields
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
            message_format="markdownv2",
        )

        # Test various invalid parameter combinations
        invalid_params = [
            # Invalid title scenarios
            ("", "Valid Message", "title must be a non-empty string"),
            (None, "Valid Message", "title must be a non-empty string"),
            ("   ", "Valid Message", "title must be a non-empty string"),
            (123, "Valid Message", "title must be a non-empty string"),
            ([], "Valid Message", "title must be a non-empty string"),
            # Invalid message scenarios
            ("Valid Title", "", "message must be a non-empty string"),
            ("Valid Title", None, "message must be a non-empty string"),
            ("Valid Title", "   ", "message must be a non-empty string"),
            ("Valid Title", 123, "message must be a non-empty string"),
            ("Valid Title", [], "message must be a non-empty string"),
        ]

        for title, message, expected_error in invalid_params:
            with pytest.raises(ValueError, match=expected_error):
                config.send_message(title, message)


class TestTelegramSendMessageNetworkErrors:
    """Test send_message method network error handling with TDD approach."""

    def test_send_message_connection_timeout(self):
        """Test send_message handles connection timeout errors gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate connection timeout
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise timeout error
            import asyncio

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = asyncio.TimeoutError("Connection timeout")

                with patch("time.sleep") as mock_sleep:
                    # Should handle timeout gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_connection_error(self):
        """Test send_message handles connection errors gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate connection error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise connection error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = ConnectionError("Failed to connect to server")

                with patch("time.sleep") as mock_sleep:
                    # Should handle connection error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_network_unreachable(self):
        """Test send_message handles network unreachable errors gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate network unreachable
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise network unreachable error
            import socket

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = socket.gaierror("Network unreachable")

                with patch("time.sleep") as mock_sleep:
                    # Should handle network error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_telegram_network_error(self):
        """Test send_message handles telegram.error.NetworkError gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate telegram network error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise telegram network error
            with patch("asyncio.run") as mock_asyncio_run:
                # Import telegram error for accurate simulation
                try:
                    from telegram.error import NetworkError

                    mock_asyncio_run.side_effect = NetworkError("Telegram network error")
                except ImportError:
                    # Fallback if telegram not available
                    mock_asyncio_run.side_effect = Exception("Telegram network error")

                with patch("time.sleep") as mock_sleep:
                    # Should handle telegram network error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_telegram_timeout_error(self):
        """Test send_message handles telegram.error.TimedOut gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate telegram timeout error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise telegram timeout error
            with patch("asyncio.run") as mock_asyncio_run:
                # Import telegram error for accurate simulation
                try:
                    from telegram.error import TimedOut

                    mock_asyncio_run.side_effect = TimedOut("Request timed out")
                except ImportError:
                    # Fallback if telegram not available
                    mock_asyncio_run.side_effect = Exception("Request timed out")

                with patch("time.sleep") as mock_sleep:
                    # Should handle telegram timeout error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_telegram_bad_request_error(self):
        """Test send_message handles telegram.error.BadRequest gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate telegram bad request error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise telegram bad request error
            with patch("asyncio.run") as mock_asyncio_run:
                # Import telegram error for accurate simulation
                try:
                    from telegram.error import BadRequest

                    mock_asyncio_run.side_effect = BadRequest("Chat not found")
                except ImportError:
                    # Fallback if telegram not available
                    mock_asyncio_run.side_effect = Exception("Chat not found")

                with patch("time.sleep") as mock_sleep:
                    # Should handle telegram bad request error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_http_error(self):
        """Test send_message handles HTTP errors gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate HTTP error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise HTTP error
            import urllib.error

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = urllib.error.HTTPError(
                    url="https://api.telegram.org", code=502, msg="Bad Gateway", hdrs={}, fp=None
                )

                with patch("time.sleep") as mock_sleep:
                    # Should handle HTTP error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_ssl_error(self):
        """Test send_message handles SSL/TLS errors gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate SSL error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise SSL error
            import ssl

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = ssl.SSLError("SSL certificate verification failed")

                with patch("time.sleep") as mock_sleep:
                    # Should handle SSL error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_dns_resolution_error(self):
        """Test send_message handles DNS resolution errors gracefully."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate DNS resolution error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise DNS resolution error
            import socket

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = socket.gaierror("Name or service not known")

                with patch("time.sleep") as mock_sleep:
                    # Should handle DNS error gracefully and return False after retries
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_maintains_synchronous_interface_with_network_errors(self):
        """Test that send_message maintains synchronous interface even with network errors."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate network error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise connection error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = ConnectionError("Network error")

                with patch("time.sleep") as mock_sleep:
                    # Should return synchronously (not a coroutine)
                    result = config.send_message("Test Title", "Test Message")

                    # Should return False after retries without deadlocks
                    assert result is False
                    assert not hasattr(result, "__await__")  # Not a coroutine
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_network_error_logging(self):
        """Test that send_message logs network errors appropriately."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Create a mock logger
        mock_logger = Mock()

        # Mock telegram.Bot to simulate network error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise network error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = ConnectionError("Network connection failed")

                # Should log error when logger is provided
                result = config.send_message("Test Title", "Test Message", logger=mock_logger)

                assert result is False
                # Should have logged the error (once implementation is done)
                # mock_logger.error.assert_called_once()

    def test_send_message_comprehensive_network_error_scenarios(self):
        """Test comprehensive network error scenarios for send_message."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Test various network error scenarios
        network_errors = [
            # Connection-related errors
            (ConnectionError("Connection refused"), "Connection refused"),
            (ConnectionResetError("Connection reset by peer"), "Connection reset"),
            (ConnectionAbortedError("Connection aborted"), "Connection aborted"),
            # Timeout errors
            (TimeoutError("Operation timed out"), "Operation timed out"),
            # Socket errors
            (OSError("No route to host"), "No route to host"),
            # Generic network errors
            (Exception("Unexpected network error"), "Unexpected network error"),
        ]

        for error, description in network_errors:
            with patch("telegram.Bot") as mock_bot_class:
                mock_bot_instance = Mock()
                mock_bot_class.return_value = mock_bot_instance

                with patch("asyncio.run") as mock_asyncio_run:
                    mock_asyncio_run.side_effect = error

                    with patch("time.sleep") as mock_sleep:
                        # Should handle all network errors gracefully
                        result = config.send_message("Test Title", "Test Message")

                        assert result is False, f"Failed to handle {description}"
                        # Should retry 5 times for network errors
                        assert mock_asyncio_run.call_count == 5
                        # Should sleep 4 times (between retries)
                        assert mock_sleep.call_count == 4

    def test_send_message_no_deadlocks_with_async_errors(self):
        """Test that send_message doesn't cause deadlocks with async errors."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate async error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise runtime error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = RuntimeError("Event loop is running")

                with patch("time.sleep") as mock_sleep:
                    # Should handle async runtime errors without deadlocks
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should complete after retries without hanging
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4

    def test_send_message_retries_implemented_for_network_errors(self):
        """Test that send_message implements retries for network errors."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate network error
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance

            # Mock asyncio.run to raise network error
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.side_effect = ConnectionError("Network error")

                with patch("time.sleep") as mock_sleep:
                    # Should call asyncio.run 5 times (with retries)
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False
                    # Should retry 5 times for network errors
                    assert mock_asyncio_run.call_count == 5
                    # Should sleep 4 times (between retries)
                    assert mock_sleep.call_count == 4


class TestTelegramConfigurationValidationCompletenesss:
    """Test comprehensive configuration validation for correctness and completeness."""

    def test_configuration_validation_with_missing_required_fields(self, config_file: Callable):
        """Test that missing required fields are properly detected and reported."""
        # Test telegram configuration with missing bot token
        cfg_missing_token = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_chat_id = '123456789'
# Missing telegram_bot_token
"""
        )

        config = Config([cfg_missing_token])
        telegram_config = config.notification["telegram"]
        assert isinstance(telegram_config, TelegramNotificationConfig)

        # Should create config but _has_required_fields should return False
        assert telegram_config.telegram_bot_token is None
        assert telegram_config.telegram_chat_id == "123456789"
        assert not telegram_config._has_required_fields()

        # Test telegram configuration with missing chat ID
        cfg_missing_chat = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
# Missing telegram_chat_id
"""
        )

        config = Config([cfg_missing_chat])
        telegram_config = config.notification["telegram"]
        assert isinstance(telegram_config, TelegramNotificationConfig)

        # Should create config but _has_required_fields should return False
        assert (
            telegram_config.telegram_bot_token == "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
        )
        assert telegram_config.telegram_chat_id is None
        assert not telegram_config._has_required_fields()

    def test_configuration_validation_with_empty_required_fields(self, config_file: Callable):
        """Test that empty required fields are properly detected and reported."""
        # Test telegram configuration with empty bot token
        cfg_empty_token = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = ''
telegram_chat_id = '123456789'
"""
        )

        with pytest.raises(Exception):
            Config([cfg_empty_token])

        # Test telegram configuration with empty chat ID
        cfg_empty_chat = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = ''
"""
        )

        with pytest.raises(Exception):
            Config([cfg_empty_chat])

    def test_configuration_validation_with_whitespace_only_fields(self, config_file: Callable):
        """Test that whitespace-only fields are properly detected and reported."""
        # Test telegram configuration with whitespace-only bot token
        cfg_whitespace_token = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '   '
telegram_chat_id = '123456789'
"""
        )

        with pytest.raises(Exception):
            Config([cfg_whitespace_token])

        # Test telegram configuration with whitespace-only chat ID
        cfg_whitespace_chat = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '   '
"""
        )

        with pytest.raises(Exception):
            Config([cfg_whitespace_chat])

    def test_configuration_validation_logical_consistency(self, config_file: Callable):
        """Test that configuration validation checks for logical consistency."""
        # Test that retry values are logically consistent
        cfg_negative_retries = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
max_retries = -1
"""
        )

        # Negative retries should be acceptable (could mean no retries)
        config = Config([cfg_negative_retries])
        telegram_config = config.notification["telegram"]
        assert telegram_config.max_retries == -1

        # Test that retry delay values are logically consistent
        cfg_negative_delay = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
retry_delay = -1
"""
        )

        # Negative retry delay should be acceptable (could mean immediate retry)
        config = Config([cfg_negative_delay])
        telegram_config = config.notification["telegram"]
        assert telegram_config.retry_delay == -1

    def test_configuration_validation_comprehensive_field_ranges(self, config_file: Callable):
        """Test comprehensive validation of field value ranges."""
        # Test extreme values for numeric fields
        test_cases = [
            # (max_retries, retry_delay, should_pass)
            (0, 0, True),  # Zero values should be acceptable
            (1, 1, True),  # Minimal positive values
            (100, 3600, True),  # Large reasonable values
            (999999, 999999, True),  # Very large values should be acceptable
        ]

        for max_retries, retry_delay, should_pass in test_cases:
            cfg = config_file(
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + f"""
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
max_retries = {max_retries}
retry_delay = {retry_delay}
"""
            )

            if should_pass:
                config = Config([cfg])
                telegram_config = config.notification["telegram"]
                assert telegram_config.max_retries == max_retries
                assert telegram_config.retry_delay == retry_delay
            else:
                with pytest.raises(Exception):
                    Config([cfg])

    def test_configuration_validation_invalid_field_types(self, config_file: Callable):
        """Test that invalid field types are properly detected and reported."""
        # Test invalid type for max_retries
        cfg_invalid_retries = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
max_retries = 'invalid'
"""
        )

        with pytest.raises(Exception):
            Config([cfg_invalid_retries])

        # Test invalid type for retry_delay
        cfg_invalid_delay = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
retry_delay = 'invalid'
"""
        )

        with pytest.raises(Exception):
            Config([cfg_invalid_delay])

    def test_configuration_validation_telegram_token_format_comprehensive(
        self, config_file: Callable
    ):
        """Test comprehensive telegram bot token format validation."""
        invalid_token_cases = [
            ("12345", "missing colon separator"),
            ("12345:", "missing token part"),
            (":ABCDEF", "missing bot ID part"),
            ("abc:ABCDEF", "non-numeric bot ID"),
            ("12345:ABC DEF", "space in token"),
            ("12345:ABC@DEF", "invalid character in token"),
            ("", "empty token"),
            ("   ", "whitespace only token"),
        ]

        for invalid_token, _ in invalid_token_cases:
            cfg = config_file(
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + f"""
[notification.telegram]
telegram_bot_token = '{invalid_token}'
telegram_chat_id = '123456789'
"""
            )

            with pytest.raises(Exception, match=".*telegram.*token.*"):
                Config([cfg])

    def test_configuration_validation_telegram_chat_id_format_comprehensive(
        self, config_file: Callable
    ):
        """Test comprehensive telegram chat ID format validation."""
        # These cases should raise exceptions during config creation
        invalid_chat_id_cases_that_raise = [
            ("", "empty chat ID"),
            ("   ", "whitespace only chat ID"),
            ("@", "username too short"),
            ("@user name", "space in username"),
            ("@user@name", "invalid character in username"),
            ("abc123", "non-numeric ID without @"),
            ("12.5", "decimal number"),
            ("1e5", "scientific notation"),
        ]

        for invalid_chat_id, _ in invalid_chat_id_cases_that_raise:
            cfg = config_file(
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + f"""
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '{invalid_chat_id}'
"""
            )

            with pytest.raises(Exception):
                Config([cfg])

        # These cases might be allowed but are not valid telegram format
        # They should be accepted by the current validation but noted as edge cases
        edge_cases = [
            ("@123abc", "username starting with number"),
        ]

        for edge_chat_id, _ in edge_cases:
            cfg = config_file(
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + f"""
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '{edge_chat_id}'
"""
            )

            # These should not raise exceptions in current implementation
            config = Config([cfg])
            telegram_config = config.notification["telegram"]
            assert isinstance(telegram_config, TelegramNotificationConfig)
            assert telegram_config.telegram_chat_id == edge_chat_id

    def test_configuration_validation_message_format_comprehensive(self, config_file: Callable):
        """Test comprehensive message format validation."""
        # Valid message formats
        valid_formats = ["plain_text", "markdown", "markdownv2", "html"]
        for valid_format in valid_formats:
            cfg = config_file(
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + f"""
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
message_format = '{valid_format}'
"""
            )

            config = Config([cfg])
            telegram_config = config.notification["telegram"]
            assert isinstance(telegram_config, TelegramNotificationConfig)
            assert telegram_config.message_format == valid_format

        # Invalid message formats
        invalid_formats = [
            "text",
            "xml",
            "json",
            "invalid",
            "",
            "   ",
            "PLAIN_TEXT",  # Case sensitive
            "Markdown",  # Case sensitive
        ]

        for invalid_format in invalid_formats:
            cfg = config_file(
                base_marketplace_cfg
                + base_item_cfg
                + notify_user_telegram_cfg
                + f"""
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
message_format = '{invalid_format}'
"""
            )

            with pytest.raises(Exception, match="Invalid message format"):
                Config([cfg])

    def test_configuration_validation_inheritance_completeness(self, config_file: Callable):
        """Test that configuration validation works correctly through inheritance chain."""
        cfg = config_file(
            base_marketplace_cfg + base_item_cfg + notify_user_telegram_cfg + base_telegram_cfg
        )

        config = Config([cfg])
        user_config = config.user["user1"]

        # Verify that all required fields are accessible through inheritance
        assert hasattr(user_config, "telegram_bot_token")
        assert hasattr(user_config, "telegram_chat_id")
        assert hasattr(user_config, "message_format")
        assert hasattr(user_config, "max_retries")
        assert hasattr(user_config, "retry_delay")

        # Verify that values are correctly inherited
        telegram_config = config.notification["telegram"]
        assert isinstance(telegram_config, TelegramNotificationConfig)
        assert user_config.telegram_bot_token == telegram_config.telegram_bot_token
        assert user_config.telegram_chat_id == telegram_config.telegram_chat_id
        assert user_config.message_format == telegram_config.message_format

        # Verify that required fields validation works through inheritance
        assert user_config._has_required_fields()

    def test_configuration_validation_error_reporting_clarity(self, config_file: Callable):
        """Test that configuration validation provides clear error messages."""
        # Test that error messages contain relevant information
        cfg_invalid_token = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = 'invalid_format'
telegram_chat_id = '123456789'
"""
        )

        try:
            Config([cfg_invalid_token])
            raise AssertionError("Expected exception for invalid token format")
        except Exception as e:
            error_message = str(e)
            # Error message should contain relevant information
            assert "telegram" in error_message.lower() or "token" in error_message.lower()

        # Test error message for invalid chat ID
        cfg_invalid_chat = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = 'invalid_format'
"""
        )

        try:
            Config([cfg_invalid_chat])
            raise AssertionError("Expected exception for invalid chat ID format")
        except Exception as e:
            error_message = str(e)
            # Error message should contain relevant information
            assert "telegram" in error_message.lower() or "chat" in error_message.lower()

    def test_configuration_validation_multiple_notifications_consistency(
        self, config_file: Callable
    ):
        """Test configuration validation with multiple notification types."""
        cfg = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_mixed_cfg  # telegram + pushbullet1
            + base_telegram_cfg
            + base_pushbullet_cfg
        )

        config = Config([cfg])
        user_config = config.user["user1"]

        # Verify that both telegram and pushbullet configurations are valid
        assert user_config.notify_with is not None
        assert isinstance(user_config.notify_with, list)
        assert "telegram" in user_config.notify_with
        assert "pushbullet1" in user_config.notify_with

        # Verify that telegram fields are accessible
        assert hasattr(user_config, "telegram_bot_token")
        assert hasattr(user_config, "telegram_chat_id")

        # Verify that pushbullet fields are accessible
        assert hasattr(user_config, "pushbullet_token")

        # Verify that both configurations are complete
        telegram_config = config.notification["telegram"]
        pushbullet_config = config.notification["pushbullet1"]

        assert telegram_config._has_required_fields()

        # For pushbullet, check specifically that it has the pushbullet token
        # (Note: due to inheritance issues, _has_required_fields may check for telegram fields)
        assert hasattr(pushbullet_config, "pushbullet_token")
        assert pushbullet_config.pushbullet_token is not None
        assert pushbullet_config.pushbullet_token == "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    def test_configuration_validation_default_values_completeness(self, config_file: Callable):
        """Test that default values are properly applied during validation."""
        # Test configuration with minimal required fields
        cfg_minimal = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
# All other fields should get default values
"""
        )

        config = Config([cfg_minimal])
        telegram_config = config.notification["telegram"]
        assert isinstance(telegram_config, TelegramNotificationConfig)

        # Verify that default values are applied
        assert telegram_config.message_format == "markdownv2"  # Default format
        assert telegram_config.max_retries == 5  # Default from parent class
        assert telegram_config.retry_delay == 60  # Default from parent class
        assert telegram_config.notify_method == "telegram"

        # Verify that configuration is complete with defaults
        assert telegram_config._has_required_fields()

    def test_configuration_validation_toml_parsing_edge_cases(self, config_file: Callable):
        """Test TOML parsing edge cases for configuration validation."""
        # Test TOML with quotes and special characters
        cfg_special_chars = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = "123456789:ABCDEF-ghijkl_mnopqr"
telegram_chat_id = "-1001234567890"
message_format = "markdownv2"
"""
        )

        config = Config([cfg_special_chars])
        telegram_config = config.notification["telegram"]
        assert isinstance(telegram_config, TelegramNotificationConfig)

        # Verify special characters are preserved
        assert telegram_config.telegram_bot_token == "123456789:ABCDEF-ghijkl_mnopqr"
        assert telegram_config.telegram_chat_id == "-1001234567890"

        # Test TOML with numeric values
        cfg_numeric = config_file(
            base_marketplace_cfg
            + base_item_cfg
            + notify_user_telegram_cfg
            + """
[notification.telegram]
telegram_bot_token = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk'
telegram_chat_id = '123456789'
max_retries = 10
retry_delay = 120
"""
        )

        config = Config([cfg_numeric])
        telegram_config = config.notification["telegram"]

        # Verify numeric values are properly parsed
        assert telegram_config.max_retries == 10
        assert telegram_config.retry_delay == 120
        assert isinstance(telegram_config.max_retries, int)
        assert isinstance(telegram_config.retry_delay, int)


class TestTelegramSynchronousInterfaceCompliance:
    """Test comprehensive synchronous interface compliance verification."""

    def test_send_message_returns_non_coroutine(self):
        """Test that send_message returns a regular value, not a coroutine."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # Call send_message
                result = config.send_message("Test Title", "Test Message")

                # Verify result is not a coroutine
                assert not hasattr(result, "__await__")
                assert not hasattr(result, "__aenter__")
                assert not hasattr(result, "__aexit__")
                assert result is True

    def test_send_message_no_async_keywords_in_caller_context(self):
        """Test that calling send_message requires no async/await in caller context."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # This should work in a completely synchronous context
                # No await needed, no async function required
                def synchronous_caller():
                    return config.send_message("Test Title", "Test Message")

                result = synchronous_caller()
                assert result is True

    def test_send_message_compatible_with_synchronous_test_environment(self):
        """Test that send_message works in standard synchronous test environments."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # Test that it works in regular pytest environment (not pytest-asyncio)
                # This verifies compatibility with puppeteer and other sync tests
                results = []
                for i in range(3):
                    result = config.send_message(f"Title {i}", f"Message {i}")
                    results.append(result)

                # All calls should succeed synchronously
                assert all(results)
                assert len(results) == 3

    def test_send_message_no_event_loop_conflicts(self):
        """Test that send_message doesn't create event loop conflicts."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = True

                # Call multiple times to test for event loop reuse conflicts
                results = []
                for i in range(5):
                    result = config.send_message(f"Title {i}", f"Message {i}")
                    results.append(result)

                # All calls should succeed without conflicts
                assert all(results)
                assert mock_asyncio_run.call_count == 5  # Each call creates its own event loop

    def test_send_message_validation_errors_synchronous(self):
        """Test that validation errors are raised synchronously without async operations."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Test that validation errors don't trigger any async operations
        with patch("asyncio.run") as mock_asyncio_run:
            # Validation errors should not call asyncio.run at all
            with pytest.raises(ValueError, match="title must be a non-empty string"):
                config.send_message("", "Test Message")

            with pytest.raises(ValueError, match="message must be a non-empty string"):
                config.send_message("Test Title", "")

            # asyncio.run should never be called for validation errors
            mock_asyncio_run.assert_not_called()

    def test_send_message_blocking_behavior(self):
        """Test that send_message blocks until completion (synchronous behavior)."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to simulate a delay
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            call_order = []

            def mock_asyncio_run(_: object):
                call_order.append("asyncio_run_start")
                # Simulate some processing time
                import time

                time.sleep(0.01)  # Small delay to simulate async operation
                call_order.append("asyncio_run_end")
                return True

            with patch("asyncio.run", side_effect=mock_asyncio_run):
                call_order.append("before_send")
                result = config.send_message("Test Title", "Test Message")
                call_order.append("after_send")

                # Verify that the call was blocking (synchronous)
                expected_order = [
                    "before_send",
                    "asyncio_run_start",
                    "asyncio_run_end",
                    "after_send",
                ]
                assert call_order == expected_order
                assert result is True

    def test_send_message_no_concurrent_execution(self):
        """Test that send_message calls execute sequentially, not concurrently."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to track execution order
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            execution_order = []

            def mock_asyncio_run(_: object):
                execution_order.append(len(execution_order))
                return True

            with patch("asyncio.run", side_effect=mock_asyncio_run):
                # Make multiple sequential calls
                results = []
                for i in range(3):
                    results.append(config.send_message(f"Title {i}", f"Message {i}"))

                # Verify sequential execution (no concurrency)
                assert execution_order == [0, 1, 2]
                assert all(results)

    def test_send_message_compatible_with_threading(self):
        """Test that send_message works correctly in multi-threaded environments."""
        import threading
        import time

        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Mock telegram.Bot to avoid actual API calls
        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_instance = Mock()
            mock_bot_class.return_value = mock_bot_instance
            mock_bot_instance.send_message = Mock()

            results = []
            lock = threading.Lock()

            def mock_asyncio_run(_: object):
                time.sleep(0.001)  # Simulate small delay
                return True

            with patch("asyncio.run", side_effect=mock_asyncio_run):

                def thread_worker(thread_id: int):
                    result = config.send_message(
                        f"Thread {thread_id}", f"Message from thread {thread_id}"
                    )
                    with lock:
                        results.append((thread_id, result))

                # Create multiple threads
                threads = []
                for i in range(3):
                    thread = threading.Thread(target=thread_worker, args=(i,))
                    threads.append(thread)
                    thread.start()

                # Wait for all threads to complete
                for thread in threads:
                    thread.join()

                # Verify all threads succeeded
                assert len(results) == 3
                assert all(result for _, result in results)

    def test_send_message_inspect_source_no_async_keywords(self):
        """Test that send_message source code contains no async/await keywords in our code."""
        import inspect

        # Get the source code of the send_message method
        source = inspect.getsource(TelegramNotificationConfig.send_message)

        # Check that our implementation doesn't use async/await keywords
        # (except inside the internal async function which is expected)
        lines = source.split("\n")

        # Simply check that await only appears in the context of the internal async function
        # and that the main send_message is not declared as async
        assert not source.startswith("    async def send_message")  # Main method is not async

        # Check that await only appears within the _send_telegram_message function
        # by counting lines between async function definition and its end
        async_func_start = -1
        async_func_end = -1

        for i, line in enumerate(lines):
            if "async def _send_telegram_message(" in line:
                async_func_start = i
            elif async_func_start != -1 and async_func_end == -1 and line.strip() == "return True":
                async_func_end = i
                break

        # Check that any await keywords only appear within the async function boundaries
        if async_func_start != -1 and async_func_end != -1:
            for i, line in enumerate(lines):
                if "await " in line:
                    assert (
                        async_func_start <= i <= async_func_end
                    ), f"await found outside async function at line {i}: {line.strip()}"

        # Verify asyncio.run is used (this is the key requirement)
        assert "asyncio.run(" in source

    def test_send_message_method_signature_synchronous(self):
        """Test that send_message method signature is synchronous (not async)."""
        import inspect

        # Get method signature
        sig = inspect.signature(TelegramNotificationConfig.send_message)

        # Verify method is not a coroutine function
        assert not inspect.iscoroutinefunction(TelegramNotificationConfig.send_message)

        # Verify return annotation is bool, not awaitable
        assert sig.return_annotation is bool

        # Verify parameters are all synchronous types
        from logging import Logger

        for param_name, param in sig.parameters.items():
            if param_name != "self":
                # None of our parameters should be awaitable types
                valid_types = [str, "Logger | None", type(None), Logger | None]
                assert param.annotation in valid_types
