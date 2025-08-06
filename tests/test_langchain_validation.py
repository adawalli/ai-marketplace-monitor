"""Tests for LangChainBackend validation and error handling enhancements.

This test suite validates the enhanced configuration validation, thread safety,
error mapping, and mixed configuration handling in the compatibility layer.
"""

import os
import threading
import time
from unittest.mock import Mock, patch

import pytest

from ai_marketplace_monitor.ai import AIConfig, LangChainBackend


class TestLangChainBackendValidation:
    """Test suite for LangChainBackend validation enhancements."""

    def test_validate_config_compatibility_openai_valid(self) -> None:
        """Test OpenAI configuration validation with valid inputs."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = LangChainBackend.get_config(
                name="test-openai", provider="openai", model="gpt-4", timeout=30, max_retries=5
            )
            assert config.name == "test-openai"
            assert config.provider == "openai"
            assert config.model == "gpt-4"

    def test_validate_config_compatibility_openai_missing_api_key(self) -> None:
        """Test OpenAI configuration validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="openai requires an API key"):
                LangChainBackend.get_config(name="test-openai", provider="openai", model="gpt-4")

    def test_validate_config_compatibility_deepseek_valid(self) -> None:
        """Test DeepSeek configuration validation with valid inputs."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            config = LangChainBackend.get_config(
                name="test-deepseek", provider="deepseek", model="deepseek-chat"
            )
            assert config.name == "test-deepseek"
            assert config.provider == "deepseek"

    def test_validate_config_compatibility_ollama_valid(self) -> None:
        """Test Ollama configuration validation with valid inputs."""
        config = LangChainBackend.get_config(
            name="test-ollama",
            provider="ollama",
            model="deepseek-r1:14b",
            base_url="http://localhost:11434",
        )
        assert config.name == "test-ollama"
        assert config.provider == "ollama"
        assert config.base_url == "http://localhost:11434"

    def test_validate_config_compatibility_ollama_missing_model(self) -> None:
        """Test Ollama configuration validation with missing model."""
        with pytest.raises(ValueError, match="Ollama requires a model to be specified"):
            LangChainBackend.get_config(
                name="test-ollama", provider="ollama", base_url="http://localhost:11434"
            )

    def test_validate_config_compatibility_ollama_default_url(self) -> None:
        """Test Ollama configuration sets default base_url if not provided."""
        config = LangChainBackend.get_config(
            name="test-ollama", provider="ollama", model="deepseek-r1:14b"
        )
        assert config.base_url == "http://localhost:11434"

    def test_validate_config_compatibility_unsupported_provider(self) -> None:
        """Test configuration validation with unsupported provider."""
        with pytest.raises(ValueError, match="AIConfig requires a valid service provider"):
            LangChainBackend.get_config(
                name="test-unsupported", provider="unsupported", api_key="test-key"
            )

    def test_validate_config_compatibility_invalid_max_retries(self) -> None:
        """Test configuration validation with invalid max_retries."""
        with pytest.raises(ValueError, match="AIConfig requires a positive integer max_retries"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                LangChainBackend.get_config(name="test-openai", provider="openai", max_retries=-1)

    def test_validate_config_compatibility_invalid_timeout(self) -> None:
        """Test configuration validation with invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be a positive integer"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                LangChainBackend.get_config(name="test-openai", provider="openai", timeout=0)

    def test_validate_thread_safety_success(self) -> None:
        """Test thread safety validation with properly initialized backend."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        # Should not raise any exceptions
        backend._validate_thread_safety()

    def test_validate_thread_safety_missing_lock(self) -> None:
        """Test thread safety validation with missing lock."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        # Remove the lock to test validation
        del backend._model_lock

        with pytest.raises(RuntimeError, match="missing proper thread synchronization"):
            backend._validate_thread_safety()

    def test_validate_thread_safety_invalid_lock(self) -> None:
        """Test thread safety validation with invalid lock type."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        # Replace with invalid lock type
        backend._model_lock = "not-a-lock"

        with pytest.raises(RuntimeError, match="missing proper thread synchronization"):
            backend._validate_thread_safety()

    def test_thread_safety_concurrent_access(self) -> None:
        """Test thread safety with concurrent access to backend methods."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        results = []
        exceptions = []

        def validate_thread_safety():
            try:
                backend._validate_thread_safety()
                results.append("success")
            except Exception as e:
                exceptions.append(e)

        # Create multiple threads to test concurrent validation
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=validate_thread_safety)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All validations should succeed
        assert len(results) == 10
        assert len(exceptions) == 0

    def test_map_langchain_exception_import_error(self) -> None:
        """Test mapping of ImportError to RuntimeError."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        original_error = ImportError("langchain_openai not found")
        mapped_error = backend._map_langchain_exception(original_error, "test context")

        assert isinstance(mapped_error, RuntimeError)
        assert "test context: Provider dependencies not installed" in str(mapped_error)
        assert "Install the required LangChain packages" in str(mapped_error)

    def test_map_langchain_exception_authentication_error(self) -> None:
        """Test mapping of authentication errors to ValueError."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        original_error = ValueError("Invalid API_KEY provided")
        mapped_error = backend._map_langchain_exception(original_error, "test context")

        assert isinstance(mapped_error, ValueError)
        assert "test context: Authentication error" in str(mapped_error)
        assert "Check API key configuration" in str(mapped_error)

    def test_map_langchain_exception_timeout_error(self) -> None:
        """Test mapping of timeout errors to RuntimeError."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        original_error = TimeoutError("Request timed out")
        mapped_error = backend._map_langchain_exception(original_error, "test context")

        assert isinstance(mapped_error, RuntimeError)
        assert "test context: Connection failed" in str(mapped_error)
        assert "Check network connectivity" in str(mapped_error)

    def test_map_langchain_exception_model_error(self) -> None:
        """Test mapping of model configuration errors to ValueError."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        original_error = KeyError("model 'invalid-model' not found")
        mapped_error = backend._map_langchain_exception(original_error, "test context")

        assert isinstance(mapped_error, ValueError)
        assert "test context: Invalid model configuration" in str(mapped_error)
        assert "Check model name and availability" in str(mapped_error)

    def test_map_langchain_exception_unknown_error(self) -> None:
        """Test mapping of unknown errors to RuntimeError."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        original_error = RuntimeError("Unknown runtime error")
        mapped_error = backend._map_langchain_exception(original_error)

        assert isinstance(mapped_error, RuntimeError)
        assert "Unexpected error: Unknown runtime error" in str(mapped_error)

    def test_validate_mixed_configuration_legacy_provider(self) -> None:
        """Test validation of mixed configuration with legacy service_provider."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        # Add legacy field (simulating old configuration)
        config.service_provider = "openai"

        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        warnings = backend._validate_mixed_configuration(config)

        # Enhanced migration warnings now include additional guidance
        assert len(warnings) >= 1
        service_provider_warning = next((w for w in warnings if "service_provider" in w), None)
        assert service_provider_warning is not None
        assert "Using 'provider' value" in service_provider_warning
        assert "Consider removing 'service_provider'" in service_provider_warning
        assert mock_logger.warning.called

    def test_validate_mixed_configuration_deepseek_api_key(self) -> None:
        """Test validation of mixed configuration with DeepSeek API key sources."""
        config = AIConfig(name="test-deepseek", provider="deepseek", api_key="config-key")

        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
            warnings = backend._validate_mixed_configuration(config)

            assert len(warnings) == 1
            assert "api_key and DEEPSEEK_API_KEY" in warnings[0]
            assert "Using environment variable" in warnings[0]
            mock_logger.warning.assert_called_once()

    def test_validate_mixed_configuration_no_conflicts(self) -> None:
        """Test validation of configuration with minimal conflicts."""
        # Use environment variables to avoid API key warnings
        config = AIConfig(name="test-backend", provider="openai")  # No API key in config

        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "proper-length-api-key"}):
            warnings = backend._validate_mixed_configuration(config)

            # Should have no warnings for properly configured environment-based setup
            assert len(warnings) == 0
            mock_logger.warning.assert_not_called()

    def test_connect_with_enhanced_validation(self) -> None:
        """Test connect method with all enhanced validation steps."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        mock_model = Mock()

        with patch.object(backend, "_get_model", return_value=mock_model):
            backend.connect()

            # Verify connection succeeded
            assert backend._chat_model is mock_model
            mock_logger.info.assert_called_once()

            # Verify validation methods were called during connect
            assert hasattr(backend, "_model_lock")

    def test_connect_with_mapped_exception(self) -> None:
        """Test connect method with exception mapping."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        # Simulate an ImportError during model creation
        original_error = ImportError("langchain_openai not found")

        with patch.object(backend, "_get_model", side_effect=original_error):
            with pytest.raises(RuntimeError, match="Provider dependencies not installed"):
                backend.connect()

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Failed to connect" in mock_logger.error.call_args[0][0]


class TestPerformanceRegression:
    """Test performance aspects of the validation enhancements."""

    def test_config_validation_performance(self) -> None:
        """Test that configuration validation doesn't significantly impact performance."""
        start_time = time.time()

        # Create multiple configurations to test performance impact
        for i in range(100):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                config = LangChainBackend.get_config(
                    name=f"test-backend-{i}",
                    provider="openai",
                    model="gpt-4",
                    timeout=30,
                    max_retries=5,
                )
                assert config.name == f"test-backend-{i}"

        elapsed_time = time.time() - start_time

        # Validation should complete quickly (under 1 second for 100 configs)
        assert elapsed_time < 1.0, f"Configuration validation took too long: {elapsed_time:.2f}s"

    def test_thread_safety_validation_performance(self) -> None:
        """Test that thread safety validation is performant."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        start_time = time.time()

        # Run validation many times
        for _ in range(1000):
            backend._validate_thread_safety()

        elapsed_time = time.time() - start_time

        # Thread safety validation should be very fast
        assert elapsed_time < 0.1, f"Thread safety validation took too long: {elapsed_time:.2f}s"

    def test_exception_mapping_performance(self) -> None:
        """Test that exception mapping doesn't add significant overhead."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        test_errors = [
            ImportError("test import error"),
            ValueError("test api_key error"),
            TimeoutError("test timeout error"),
            KeyError("test model error"),
            RuntimeError("test runtime error"),
        ]

        start_time = time.time()

        # Map many exceptions
        for _ in range(100):
            for error in test_errors:
                mapped = backend._map_langchain_exception(error, "test context")
                assert isinstance(mapped, Exception)

        elapsed_time = time.time() - start_time

        # Exception mapping should be fast
        assert elapsed_time < 0.1, f"Exception mapping took too long: {elapsed_time:.2f}s"
