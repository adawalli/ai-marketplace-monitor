"""Test monitor integration with LangChain backend compatibility layer."""

from src.ai_marketplace_monitor.ai import LangChainBackend
from src.ai_marketplace_monitor.config import supported_ai_backends


class TestMonitorIntegration:
    """Test that monitor.py works correctly with the LangChain backend compatibility layer."""

    def test_supported_ai_backends_uses_langchain(self) -> None:
        """Test that all supported AI backends now use LangChainBackend."""
        for provider, backend_class in supported_ai_backends.items():
            assert (
                backend_class is LangChainBackend
            ), f"Provider {provider} should use LangChainBackend"

    def test_monitor_backend_selection_works(self) -> None:
        """Test that monitor backend selection logic works with compatibility layer."""
        from src.ai_marketplace_monitor.ai import AIConfig

        # Test provider-based selection
        ai_config = AIConfig(name="test-openai", provider="openai", api_key="test-key")
        provider_key = ai_config.provider.lower() if ai_config.provider else None

        if provider_key and provider_key in supported_ai_backends:
            ai_class = supported_ai_backends[provider_key]
            assert ai_class is LangChainBackend

        # Test name-based fallback selection
        ai_config_name = AIConfig(name="deepseek", api_key="test-key")
        name_key = ai_config_name.name.lower()

        if name_key in supported_ai_backends:
            ai_class = supported_ai_backends[name_key]
            assert ai_class is LangChainBackend

    def test_case_insensitive_provider_matching(self) -> None:
        """Test that case-insensitive provider matching works through compatibility layer."""
        # Test that monitor logic handles case-insensitive matching
        test_providers = ["OpenAI", "DEEPSEEK", "ollama", "OpenRouter"]

        for provider in test_providers:
            provider_lower = provider.lower()
            assert provider_lower in supported_ai_backends
            assert supported_ai_backends[provider_lower] is LangChainBackend
