from unittest.mock import Mock, patch

import pytest

from ai_marketplace_monitor.ai import AIConfig, LangChainBackend, OllamaBackend, OllamaConfig
from ai_marketplace_monitor.facebook import FacebookItemConfig, FacebookMarketplaceConfig
from ai_marketplace_monitor.listing import Listing


@pytest.mark.skipif(True, reason="Condition met, skipping this test")
def test_ai(
    ollama_config: OllamaConfig,
    item_config: FacebookItemConfig,
    marketplace_config: FacebookMarketplaceConfig,
    listing: Listing,
) -> None:
    ai = OllamaBackend(ollama_config)
    # ai.config = ollama_config
    res = ai.evaluate(listing, item_config, marketplace_config)
    assert res.score >= 1 and res.score <= 5


def test_prompt(
    ollama: OllamaBackend,
    listing: Listing,
    item_config: FacebookItemConfig,
    marketplace_config: FacebookMarketplaceConfig,
) -> None:
    prompt = ollama.get_prompt(listing, item_config, marketplace_config)
    assert item_config.name in prompt
    assert (item_config.description or "something weird") in prompt
    assert str(item_config.min_price) in prompt
    assert str(item_config.max_price) in prompt

    assert listing.title in prompt
    assert listing.condition in prompt
    assert listing.price in prompt
    assert listing.post_url in prompt


def test_extra_prompt(
    ollama: OllamaBackend,
    listing: Listing,
    item_config: FacebookItemConfig,
    marketplace_config: FacebookMarketplaceConfig,
) -> None:
    marketplace_config.extra_prompt = "This is an extra prompt"
    prompt = ollama.get_prompt(listing, item_config, marketplace_config)
    assert "extra prompt" in prompt
    #
    item_config.extra_prompt = "This overrides marketplace prompt"
    prompt = ollama.get_prompt(listing, item_config, marketplace_config)
    assert "extra prompt" not in prompt
    assert "overrides marketplace prompt" in prompt
    #
    assert "Great deal: Fully matches" in prompt
    item_config.rating_prompt = "something else"
    prompt = ollama.get_prompt(listing, item_config, marketplace_config)
    assert "Great deal: Fully matches" not in prompt
    assert "something else" in prompt
    #
    assert "Evaluate how well this listing" in prompt
    marketplace_config.prompt = "myprompt"
    prompt = ollama.get_prompt(listing, item_config, marketplace_config)
    assert "Evaluate how well this listing" not in prompt
    assert "myprompt" in prompt


class TestLangChainBackend:
    """Test suite for LangChainBackend class."""

    def test_get_config(self) -> None:
        """Test LangChainBackend.get_config returns AIConfig."""
        config = LangChainBackend.get_config(
            name="test-backend",
            provider="openai",
            api_key="test-key",
            model="gpt-4",
        )
        assert isinstance(config, AIConfig)
        assert config.name == "test-backend"
        assert config.provider == "openai"
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"

    def test_init(self) -> None:
        """Test LangChainBackend initialization."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        assert backend.config == config
        assert backend._chat_model is None
        assert backend.logger is None

    def test_get_model_success(self) -> None:
        """Test _get_model with valid provider."""
        config = AIConfig(
            name="test-backend", provider="openai", api_key="test-key", model="gpt-4"
        )
        backend = LangChainBackend(config)

        with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
            mock_factory = Mock()
            mock_model = Mock()
            mock_factory.return_value = mock_model
            mock_provider_map.__getitem__.return_value = mock_factory
            mock_provider_map.__contains__.return_value = True
            mock_provider_map.keys.return_value = ["openai", "deepseek"]

            result = backend._get_model(config)

            assert result == mock_model
            mock_factory.assert_called_once_with(config)

    def test_get_model_no_provider(self) -> None:
        """Test _get_model with missing provider."""
        config = AIConfig(name="test-backend", api_key="test-key")
        backend = LangChainBackend(config)

        with pytest.raises(ValueError, match="AIConfig must have a provider specified"):
            backend._get_model(config)

    def test_get_model_unsupported_provider(self) -> None:
        """Test _get_model with unsupported provider."""
        # Skip provider validation by mocking it
        with patch.object(AIConfig, "handle_provider"):
            config = AIConfig(name="test-backend", provider="unsupported", api_key="test-key")
            backend = LangChainBackend(config)

            with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
                mock_provider_map.__contains__.return_value = False
                mock_provider_map.keys.return_value = ["openai", "deepseek"]

                with pytest.raises(ValueError, match="Unsupported provider 'unsupported'"):
                    backend._get_model(config)

    def test_get_model_factory_error(self) -> None:
        """Test _get_model when factory raises exception."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
            mock_factory = Mock(side_effect=Exception("Factory error"))
            mock_provider_map.__getitem__.return_value = mock_factory
            mock_provider_map.__contains__.return_value = True

            with pytest.raises(RuntimeError, match="Failed to create model for provider 'openai'"):
                backend._get_model(config)

    def test_connect_success(self) -> None:
        """Test successful connection."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        mock_model = Mock()
        with patch.object(backend, "_get_model", return_value=mock_model):
            backend.connect()

            assert backend._chat_model == mock_model
            mock_logger.info.assert_called_once()

    def test_connect_already_connected(self) -> None:
        """Test connect when already connected."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        mock_model = Mock()
        backend._chat_model = mock_model

        with patch.object(backend, "_get_model") as mock_get_model:
            backend.connect()
            mock_get_model.assert_not_called()

    def test_connect_error(self) -> None:
        """Test connection error handling."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        with patch.object(backend, "_get_model", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                backend.connect()

            mock_logger.error.assert_called_once()

    def test_evaluate_with_cache(
        self,
        item_config: FacebookItemConfig,
        marketplace_config: FacebookMarketplaceConfig,
        listing: Listing,
    ) -> None:
        """Test evaluate method with cached response."""
        config = AIConfig(provider="openai", api_key="test-key", name="test-backend")
        backend = LangChainBackend(config)

        # Mock cached response
        with patch("ai_marketplace_monitor.ai.AIResponse.from_cache") as mock_cache:
            mock_response = Mock()
            mock_response.style = "succ"
            mock_response.conclusion = "Good match"
            mock_response.score = 4
            mock_response.comment = "Test comment"
            mock_cache.return_value = mock_response

            result = backend.evaluate(listing, item_config, marketplace_config)

            assert result == mock_response

    def test_evaluate_success(
        self,
        item_config: FacebookItemConfig,
        marketplace_config: FacebookMarketplaceConfig,
        listing: Listing,
    ) -> None:
        """Test successful evaluation without cache."""
        config = AIConfig(
            provider="openai", api_key="test-key", name="test-backend", max_retries=3
        )
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        # Mock LangChain model response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "This is a good match.\nRating 4: Great condition and price"
        mock_model.invoke.return_value = mock_response

        backend._chat_model = mock_model

        with (
            patch("ai_marketplace_monitor.ai.AIResponse.from_cache", return_value=None),
            patch("ai_marketplace_monitor.ai.counter"),
            patch.object(backend, "get_prompt", return_value="test prompt"),
        ):
            result = backend.evaluate(listing, item_config, marketplace_config)

            assert result.score == 4
            assert "Great condition and price" in result.comment
            assert result.name == "test-backend"

    def test_evaluate_retry_on_error(
        self,
        item_config: FacebookItemConfig,
        marketplace_config: FacebookMarketplaceConfig,
        listing: Listing,
    ) -> None:
        """Test retry logic on evaluation error."""
        config = AIConfig(
            name="test-backend", provider="openai", api_key="test-key", max_retries=3
        )
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        # Mock connection to reset _chat_model back to mock_model after connect() is called
        def mock_connect():
            backend._chat_model = mock_model

        mock_model = Mock()
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.content = "Rating 3: Decent match"
        mock_model.invoke.side_effect = [Exception("Network error"), mock_response]

        with (
            patch("ai_marketplace_monitor.ai.AIResponse.from_cache", return_value=None),
            patch("ai_marketplace_monitor.ai.counter"),
            patch.object(backend, "connect", side_effect=mock_connect),
            patch.object(backend, "get_prompt", return_value="test prompt"),
            patch("time.sleep"),
        ):
            backend._chat_model = mock_model
            result = backend.evaluate(listing, item_config, marketplace_config)

            assert result.score == 3
            assert mock_logger.error.called

    def test_extract_response_content_with_string_content(self) -> None:
        """Test _extract_response_content with string content attribute."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        mock_response = Mock()
        mock_response.content = "Test response content"

        result = backend._extract_response_content(mock_response)
        assert result == "Test response content"

    def test_extract_response_content_with_list_content(self) -> None:
        """Test _extract_response_content with list content attribute."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        mock_response = Mock()
        mock_response.content = ["First message", "Second message"]

        result = backend._extract_response_content(mock_response)
        assert "First message" in result

    def test_extract_response_content_with_text_fallback(self) -> None:
        """Test _extract_response_content falling back to text attribute."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        mock_response = Mock()
        del mock_response.content  # Remove content attribute
        mock_response.text = "Response from text attribute"

        result = backend._extract_response_content(mock_response)
        assert result == "Response from text attribute"

    def test_extract_response_content_with_message_fallback(self) -> None:
        """Test _extract_response_content falling back to message.content."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        mock_response = Mock()
        del mock_response.content
        del mock_response.text
        mock_response.message = Mock()
        mock_response.message.content = "Message content"

        result = backend._extract_response_content(mock_response)
        assert result == "Message content"

    def test_extract_response_content_string_fallback(self) -> None:
        """Test _extract_response_content final string fallback."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        mock_response = "Raw string response"

        result = backend._extract_response_content(mock_response)
        assert result == "Raw string response"
        mock_logger.warning.assert_called_once()

    def test_evaluate_retry_with_connection_failure(
        self,
        item_config: FacebookItemConfig,
        marketplace_config: FacebookMarketplaceConfig,
        listing: Listing,
    ) -> None:
        """Test retry logic handles connection failures during reconnect attempts."""
        config = AIConfig(
            name="test-backend", provider="openai", api_key="test-key", max_retries=3
        )
        mock_logger = Mock()
        backend = LangChainBackend(config, logger=mock_logger)

        mock_model = Mock()
        # First call fails, connection during retry also fails, then succeeds
        mock_response = Mock()
        mock_response.content = "Rating 4: Good match"
        mock_model.invoke.side_effect = [
            Exception("Network error"),  # First attempt fails
            mock_response,  # Second attempt succeeds
        ]

        # Mock connect to fail on second call (during retry), then succeed
        connect_call_count = 0

        def mock_connect():
            nonlocal connect_call_count
            connect_call_count += 1
            if connect_call_count == 2:
                # Reconnect attempt during retry fails
                raise Exception("Connection failed")
            else:
                # Initial connect and final reconnect succeed
                backend._chat_model = mock_model

        with (
            patch("ai_marketplace_monitor.ai.AIResponse.from_cache", return_value=None),
            patch("ai_marketplace_monitor.ai.counter"),
            patch.object(backend, "connect", side_effect=mock_connect),
            patch.object(backend, "get_prompt", return_value="test prompt"),
            patch("time.sleep"),
        ):
            backend._chat_model = mock_model
            result = backend.evaluate(listing, item_config, marketplace_config)

            assert result.score == 4
            # Should log both the invoke error and the connection error
            assert mock_logger.error.call_count >= 2

    def test_thread_safety_with_concurrent_access(
        self,
        item_config: FacebookItemConfig,
        marketplace_config: FacebookMarketplaceConfig,
        listing: Listing,
    ) -> None:
        """Test thread safety when multiple threads access the backend concurrently."""
        import concurrent.futures
        from unittest.mock import patch

        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        mock_response = Mock()
        mock_response.content = "Rating 3: Thread-safe response"

        results = []
        errors = []

        def evaluate_listing():
            try:
                with (
                    patch("ai_marketplace_monitor.ai.AIResponse.from_cache", return_value=None),
                    patch("ai_marketplace_monitor.ai.counter"),
                    patch.object(backend, "get_prompt", return_value="test prompt"),
                    patch.object(backend, "_get_model") as mock_get_model,
                ):
                    mock_model = Mock()
                    mock_model.invoke.return_value = mock_response
                    mock_get_model.return_value = mock_model

                    result = backend.evaluate(listing, item_config, marketplace_config)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(evaluate_listing) for _ in range(10)]
            concurrent.futures.wait(futures)

        # All evaluations should succeed
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 10
        # All results should have the same score
        assert all(result.score == 3 for result in results)

    def test_parse_ai_response_basic(self) -> None:
        """Test _parse_ai_response with basic rating format."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)
        item_config = Mock()
        item_config.name = "test-item"

        answer = "This is a good match.\nRating 4: Great condition and price"

        with patch("ai_marketplace_monitor.ai.counter"):
            result = backend._parse_ai_response(answer, item_config)

        assert result.score == 4
        assert "Great condition and price" in result.comment
        assert result.name == "test-backend"

    def test_parse_ai_response_multiline_comment(self) -> None:
        """Test _parse_ai_response with multiline comment after rating."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)
        item_config = Mock()
        item_config.name = "test-item"

        answer = """This looks good.
Rating 3: Decent match
The price is reasonable
But condition could be better"""

        with patch("ai_marketplace_monitor.ai.counter"):
            result = backend._parse_ai_response(answer, item_config)

        assert result.score == 3
        assert (
            "Decent match The price is reasonable But condition could be better" in result.comment
        )

    def test_parse_ai_response_invalid_format(self) -> None:
        """Test _parse_ai_response with invalid response format."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)
        item_config = Mock()
        item_config.name = "test-item"

        answer = "No rating provided in this response"

        with patch("ai_marketplace_monitor.ai.counter") as mock_counter:
            with pytest.raises(ValueError, match="Empty or invalid response"):
                backend._parse_ai_response(answer, item_config)

            mock_counter.increment.assert_called_once()

    def test_get_model_keyerror_handling(self) -> None:
        """Test enhanced error handling for KeyError from provider factory."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
            mock_factory = Mock(side_effect=KeyError("api_key"))
            mock_provider_map.__getitem__.return_value = mock_factory
            mock_provider_map.__contains__.return_value = True

            with pytest.raises(ValueError, match="configuration error.*Check API key"):
                backend._get_model(config)

    def test_get_model_typeerror_handling(self) -> None:
        """Test enhanced error handling for TypeError from provider factory."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
            mock_factory = Mock(side_effect=TypeError("invalid parameter type"))
            mock_provider_map.__getitem__.return_value = mock_factory
            mock_provider_map.__contains__.return_value = True

            with pytest.raises(ValueError, match="configuration error.*Check API key"):
                backend._get_model(config)

    def test_get_model_importerror_handling(self) -> None:
        """Test enhanced error handling for ImportError from provider factory."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
            mock_factory = Mock(side_effect=ImportError("langchain_openai not found"))
            mock_provider_map.__getitem__.return_value = mock_factory
            mock_provider_map.__contains__.return_value = True

            with pytest.raises(
                RuntimeError, match="dependencies not installed.*Install the required"
            ):
                backend._get_model(config)

    def test_get_model_chained_exception_handling(self) -> None:
        """Test enhanced error handling for RuntimeError exceptions."""
        config = AIConfig(name="test-backend", provider="openai", api_key="test-key")
        backend = LangChainBackend(config)

        # Create a chained exception
        try:
            raise ValueError("original error")
        except ValueError as original:
            chained_error = RuntimeError("chained error")
            chained_error.__cause__ = original

            with patch("ai_marketplace_monitor.ai.provider_map") as mock_provider_map:
                mock_factory = Mock(side_effect=chained_error)
                mock_provider_map.__getitem__.return_value = mock_factory
                mock_provider_map.__contains__.return_value = True

                with pytest.raises(RuntimeError, match="Failed to create model.*chained error"):
                    backend._get_model(config)
