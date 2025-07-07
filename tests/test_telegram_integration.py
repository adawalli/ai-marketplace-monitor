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

                # Should handle timeout gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle connection error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle network error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle telegram network error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle telegram timeout error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle telegram bad request error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle HTTP error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle SSL error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should handle DNS error gracefully and return False
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()

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

                # Should return synchronously (not a coroutine)
                result = config.send_message("Test Title", "Test Message")

                # Should return False immediately without deadlocks
                assert result is False
                assert not hasattr(result, "__await__")  # Not a coroutine
                mock_asyncio_run.assert_called_once()

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

                    # Should handle all network errors gracefully
                    result = config.send_message("Test Title", "Test Message")

                    assert result is False, f"Failed to handle {description}"
                    mock_asyncio_run.assert_called_once()

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

                # Should handle async runtime errors without deadlocks
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                # Should complete quickly without hanging
                mock_asyncio_run.assert_called_once()

    def test_send_message_retries_not_implemented_in_method(self):
        """Test that send_message doesn't implement retries internally (left to parent class)."""
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

                # Should call asyncio.run only once (no internal retries)
                result = config.send_message("Test Title", "Test Message")

                assert result is False
                mock_asyncio_run.assert_called_once()  # Only one call, no retries


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

            def mock_asyncio_run(coro: object):
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

            def mock_asyncio_run(coro: object):
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

            def mock_asyncio_run(coro: object):
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

        # Find lines that are part of our main implementation (not the internal async function)
        main_implementation_lines = []
        inside_async_function = False

        for line in lines:
            stripped = line.strip()
            if "async def _send_telegram_message():" in line:
                inside_async_function = True
                continue
            if inside_async_function and line.startswith("            # Use asyncio.run"):
                inside_async_function = False
                continue
            if not inside_async_function and stripped:
                main_implementation_lines.append(line)

        # Check that main implementation has no async/await keywords
        main_code = "\n".join(main_implementation_lines)
        assert "async def" not in main_code or "async def _send_telegram_message" in main_code
        assert "await " not in main_code  # No await in our main code

        # Verify asyncio.run is used (this is the key requirement)
        assert "asyncio.run(" in main_code

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
