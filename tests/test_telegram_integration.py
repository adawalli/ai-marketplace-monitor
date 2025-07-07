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
        """Test that send_message method exists and raises NotImplementedError."""
        config = TelegramNotificationConfig(
            name="test_telegram",
            telegram_bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            telegram_chat_id="123456789",
        )

        # Method should exist
        assert hasattr(config, "send_message")
        assert callable(config.send_message)

        # But should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            config.send_message("Test Title", "Test Message")


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

        # Test that send_message method exists but is not implemented
        assert hasattr(config, "send_message")
        with pytest.raises(NotImplementedError):
            config.send_message("Test", "Message")

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
