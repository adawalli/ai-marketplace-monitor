import html
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from logging import Logger
from typing import Any, ClassVar, Generic, List, Optional, Type, TypeVar

from diskcache import Cache  # type: ignore
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI  # type: ignore
from pydantic import SecretStr
from rich.pretty import pretty_repr

from .listing import Listing
from .marketplace import TItemConfig, TMarketplaceConfig
from .utils import BaseConfig, CacheType, CounterItem, cache, counter, hilight


class AIServiceProvider(Enum):
    OPENAI = "OpenAI"
    DEEPSEEK = "DeepSeek"
    OLLAMA = "Ollama"
    OPENROUTER = "OpenRouter"


@dataclass
class AIResponse:
    score: int
    comment: str
    name: str = ""

    NOT_EVALUATED: ClassVar = "Not evaluated by AI"

    @property
    def conclusion(self: "AIResponse") -> str:
        return {
            1: "No match",
            2: "Potential match",
            3: "Poor match",
            4: "Good match",
            5: "Great deal",
        }[self.score]

    @property
    def style(self: "AIResponse") -> str:
        if self.comment == self.NOT_EVALUATED:
            return "dim"
        if self.score < 3:
            return "fail"
        if self.score > 3:
            return "succ"
        return "name"

    @property
    def stars(self: "AIResponse") -> str:
        full_stars = self.score
        empty_stars = 5 - full_stars
        return (
            '<span style="color: #FFD700; font-size: 20px;">★</span>' * full_stars
            + '<span style="color: #D3D3D3; font-size: 20px;">☆</span>' * empty_stars
        )

    @classmethod
    def from_cache(
        cls: Type["AIResponse"],
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
        local_cache: Cache | None = None,
    ) -> Optional["AIResponse"]:
        res = (cache if local_cache is None else local_cache).get(
            (CacheType.AI_INQUIRY.value, item_config.hash, marketplace_config.hash, listing.hash)
        )
        if res is None:
            return None
        return AIResponse(**res)

    def to_cache(
        self: "AIResponse",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
        local_cache: Cache | None = None,
    ) -> None:
        (cache if local_cache is None else local_cache).set(
            (CacheType.AI_INQUIRY.value, item_config.hash, marketplace_config.hash, listing.hash),
            asdict(self),
            tag=CacheType.AI_INQUIRY.value,
        )


@dataclass
class AIConfig(BaseConfig):
    # this argument is required

    api_key: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    max_retries: int = 10
    timeout: int | None = None

    def handle_provider(self: "AIConfig") -> None:
        if self.provider is None:
            return
        if self.provider.lower() not in [x.value.lower() for x in AIServiceProvider]:
            raise ValueError(
                f"""AIConfig requires a valid service provider. Valid providers are {hilight(", ".join([x.value for x in AIServiceProvider]))}"""
            )

    def handle_api_key(self: "AIConfig") -> None:
        if self.api_key is None:
            return
        if not isinstance(self.api_key, str):
            raise ValueError("AIConfig requires a string api_key.")
        self.api_key = self.api_key.strip()

    def handle_max_retries(self: "AIConfig") -> None:
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("AIConfig requires a positive integer max_retries.")

    def handle_timeout(self: "AIConfig") -> None:
        if self.timeout is None:
            return
        if not isinstance(self.timeout, int) or self.timeout < 0:
            raise ValueError("AIConfig requires a positive integer timeout.")


@dataclass
class OpenAIConfig(AIConfig):
    def handle_api_key(self: "OpenAIConfig") -> None:
        if self.api_key is None:
            raise ValueError("OpenAI requires a string api_key.")


@dataclass
class DeekSeekConfig(OpenAIConfig):
    pass


@dataclass
class OllamaConfig(OpenAIConfig):
    api_key: str | None = field(default="ollama")  # required but not used.

    def handle_base_url(self: "OllamaConfig") -> None:
        if self.base_url is None:
            raise ValueError("Ollama requires a string base_url.")

    def handle_model(self: "OllamaConfig") -> None:
        if self.model is None:
            raise ValueError("Ollama requires a string model.")


# System message for marketplace evaluation
MARKETPLACE_EVALUATION_SYSTEM_MESSAGE = """You are a marketplace listing evaluation expert. Your role is to assess how well Facebook Marketplace listings match user search criteria.

You should evaluate listings based on:
- Relevance to search terms and description
- Price reasonability vs. market value
- Item condition and seller credibility
- Completeness of listing information

Always conclude your evaluation with "Rating X:" where X is 1-5, followed by a brief summary (max 30 words).

Rating Scale:
1 - No match: Missing key details, wrong category/brand, or suspicious activity
2 - Potential match: Lacks essential info; needs clarification
3 - Poor match: Some mismatches or missing details; acceptable but not ideal
4 - Good match: Mostly meets criteria with clear, relevant details
5 - Great deal: Fully matches criteria, with excellent condition or price"""

# Few-shot examples for consistent rating behavior
FEW_SHOT_EXAMPLES = [
    {
        "user_criteria": "Looking for: iPhone 12, budget: $300-500, good condition",
        "listing": "iPhone 12 Pro 128GB, excellent condition, $450, includes charger and case",
        "evaluation": "This listing closely matches your criteria. The iPhone 12 Pro is an upgrade from the standard iPhone 12, priced within your budget at $450. Excellent condition and includes accessories add value. Seller provides clear details and photos.\n\nRating 5: Perfect match - iPhone 12 Pro in excellent condition, great price with accessories",
    },
    {
        "user_criteria": "Looking for: MacBook Air, budget: $800-1200, for college work",
        "listing": "MacBook Pro 2019 16-inch, some wear on corners, $900, works fine",
        "evaluation": "This is a MacBook Pro rather than the MacBook Air you requested. While the price fits your budget, the 16-inch Pro is heavier and more powerful than needed for typical college work. 'Some wear' and 'works fine' lack detail about actual condition.\n\nRating 2: Wrong model and vague condition description, needs more details",
    },
    {
        "user_criteria": "Looking for: Gaming chair, budget: under $200, ergonomic features",
        "listing": "IKEA office chair, used 6 months, $50, comfortable for long hours",
        "evaluation": "This is a basic office chair rather than a gaming chair with ergonomic features. However, the low price of $50 is well within budget and the seller mentions comfort for extended use. The IKEA brand is reliable.\n\nRating 3: Not specifically a gaming chair but budget-friendly office alternative",
    },
]

TAIConfig = TypeVar("TAIConfig", bound=AIConfig)


class AIBackend(Generic[TAIConfig]):
    def __init__(self: "AIBackend", config: AIConfig, logger: Logger | None = None) -> None:
        self.config = config
        self.logger = logger
        self.client: OpenAI | None = None

    @classmethod
    def get_config(cls: Type["AIBackend"], **kwargs: Any) -> TAIConfig:
        raise NotImplementedError("get_config method must be implemented by subclasses.")

    def connect(self: "AIBackend") -> None:
        raise NotImplementedError("Connect method must be implemented by subclasses.")

    def _sanitize_input(self, text: str) -> str:
        """Basic input sanitization for backwards compatibility."""
        if not text:
            return text
        # Simple HTML escape and basic pattern filtering
        return html.escape(text, quote=True)

    def get_prompt(
        self: "AIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> str:
        # Build user criteria section with sanitized inputs
        user_criteria = f"Looking for: {item_config.name}"
        if item_config.search_phrases:
            # Sanitize search terms to prevent injection
            sanitized_phrases = [
                self._sanitize_input(phrase) for phrase in item_config.search_phrases
            ]
            search_terms = '" and "'.join(sanitized_phrases)
            user_criteria += f", search terms: {search_terms}"
        if item_config.description:
            sanitized_desc = self._sanitize_input(item_config.description)
            user_criteria += f", description: {sanitized_desc}"

        # Add price constraints
        max_price = item_config.max_price or 0
        min_price = item_config.min_price or 0
        if max_price and min_price:
            user_criteria += f", budget: ${min_price}-{max_price}"
        elif max_price:
            user_criteria += f", max budget: ${max_price}"
        elif min_price:
            user_criteria += f", min budget: ${min_price}"

        # Add exclusions with sanitized inputs
        if item_config.antikeywords:
            sanitized_antikeywords = [
                self._sanitize_input(keyword) for keyword in item_config.antikeywords
            ]
            exclude_terms = '" and "'.join(sanitized_antikeywords)
            user_criteria += f", exclude: {exclude_terms}"

        # Build listing details section with sanitized inputs
        sanitized_title = self._sanitize_input(listing.title)
        sanitized_condition = self._sanitize_input(listing.condition)
        sanitized_price = self._sanitize_input(listing.price)
        sanitized_location = self._sanitize_input(listing.location)
        sanitized_description = self._sanitize_input(listing.description)

        listing_details = (
            f"Title: {sanitized_title}\n"
            f"Condition: {sanitized_condition}\n"
            f"Price: {sanitized_price}\n"
            f"Location: {sanitized_location}\n"
            f"Description: {sanitized_description}\n"
            f"URL: {listing.post_url}"
        )

        # Build evaluation instructions
        evaluation_instructions = ""
        if item_config.prompt is not None:
            evaluation_instructions = item_config.prompt
        elif marketplace_config.prompt is not None:
            evaluation_instructions = marketplace_config.prompt
        else:
            evaluation_instructions = (
                "Evaluate how well this listing matches the user's criteria. "
                "Assess the description, market value, condition, and seller credibility."
            )

        # Add extra instructions if provided
        if item_config.extra_prompt is not None:
            evaluation_instructions += f"\n\n{item_config.extra_prompt.strip()}"
        elif marketplace_config.extra_prompt is not None:
            evaluation_instructions += f"\n\n{marketplace_config.extra_prompt.strip()}"

        # Add rating instructions (for backward compatibility)
        rating_instructions = ""
        if item_config.rating_prompt is not None:
            rating_instructions = item_config.rating_prompt.strip()
        elif marketplace_config.rating_prompt is not None:
            rating_instructions = marketplace_config.rating_prompt.strip()
        else:
            rating_instructions = (
                "Rate from 1 to 5 based on the following:\n"
                "1 - No match: Missing key details, wrong category/brand, or suspicious activity.\n"
                "2 - Potential match: Lacks essential info; needs clarification.\n"
                "3 - Poor match: Some mismatches or missing details; acceptable but not ideal.\n"
                "4 - Good match: Mostly meets criteria with clear, relevant details.\n"
                "5 - Great deal: Fully matches criteria, with excellent condition or price.\n"
                'Conclude with: "Rating <1-5>: <summary>" where <1-5> is the rating and <summary> is a brief recommendation (max 30 words).'
            )

        # Build the complete prompt using structured format
        prompt = (
            f"USER CRITERIA:\n{user_criteria}\n\n"
            f"LISTING DETAILS:\n{listing_details}\n\n"
            f"EVALUATION TASK:\n{evaluation_instructions}\n\n"
            f"EXAMPLES OF GOOD EVALUATIONS:\n"
            f"Example 1: {FEW_SHOT_EXAMPLES[0]['user_criteria']}\n"
            f"Listing: {FEW_SHOT_EXAMPLES[0]['listing']}\n"
            f"Evaluation: {FEW_SHOT_EXAMPLES[0]['evaluation']}\n\n"
            f"Example 2: {FEW_SHOT_EXAMPLES[1]['user_criteria']}\n"
            f"Listing: {FEW_SHOT_EXAMPLES[1]['listing']}\n"
            f"Evaluation: {FEW_SHOT_EXAMPLES[1]['evaluation']}\n\n"
            f"RATING INSTRUCTIONS:\n{rating_instructions}\n\n"
            f"Now evaluate the listing above following the same format and rating scale."
        )

        if self.logger:
            self.logger.debug(f"""{hilight("[AI-Prompt]", "info")} {prompt}""")
        return prompt

    def _parse_ai_response(self, answer: str, item_config: TItemConfig) -> AIResponse:
        """Parse AI response text to extract rating and comment.

        This method contains the common response parsing logic used by all backends.
        """
        if (
            answer is None
            or not answer.strip()
            or re.search(r"Rating[^1-5]*[1-5]", answer, re.DOTALL) is None
        ):
            counter.increment(CounterItem.FAILED_AI_QUERY, item_config.name)
            raise ValueError(f"Empty or invalid response from {self.config.name}: {answer}")

        lines = answer.split("\n")
        # Extract rating from response
        score: int = 1
        comment = ""
        rating_line = None
        for idx, line in enumerate(lines):
            matched = re.match(r".*Rating[^1-5]*([1-5])[:\s]*(.*)", line)
            if matched:
                score = int(matched.group(1))
                comment = matched.group(2).strip()
                rating_line = idx
                continue
            if rating_line is not None:
                # if the AI puts comment after Rating, we need to include them
                comment += " " + line

        # if the AI puts the rating at the end, let us try to use the line before the Rating line
        if len(comment.strip()) < 5 and rating_line is not None and rating_line > 0:
            comment = lines[rating_line - 1]

        # remove multiple spaces, take first 30 words
        comment = " ".join([x for x in comment.split() if x.strip()]).strip()
        res = AIResponse(name=self.config.name, score=score, comment=comment)
        return res

    def evaluate(
        self: "AIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> AIResponse:
        raise NotImplementedError("Confirm method must be implemented by subclasses.")


class OpenAIBackend(AIBackend):
    default_model = "gpt-4o"
    # the default is f"https://api.openai.com/v1"
    base_url: str | None = None

    @classmethod
    def get_config(cls: Type["OpenAIBackend"], **kwargs: Any) -> OpenAIConfig:
        return OpenAIConfig(**kwargs)

    def connect(self: "OpenAIBackend") -> None:
        if self.client is None:
            # Ensure API key is properly converted to string for OpenAI client
            api_key = self.config.api_key
            if api_key is None:
                raise ValueError(f"API key is required for {self.config.provider}")
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url or self.base_url,
                timeout=self.config.timeout,
                default_headers={
                    "X-Title": "AI Marketplace Monitor",
                    "HTTP-Referer": "https://github.com/BoPeng/ai-marketplace-monitor",
                },
            )
            if self.logger:
                self.logger.info(f"""{hilight("[AI]", "name")} {self.config.name} connected.""")

    def evaluate(
        self: "OpenAIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> AIResponse:
        # ask openai to confirm the item is correct
        counter.increment(CounterItem.AI_QUERY, item_config.name)
        prompt = self.get_prompt(listing, item_config, marketplace_config)
        res: AIResponse | None = AIResponse.from_cache(listing, item_config, marketplace_config)
        if res is not None:
            if self.logger:
                self.logger.debug(
                    f"""{hilight("[AI]", res.style)} {self.config.name} previously concluded {hilight(f"{res.conclusion} ({res.score}): {res.comment}", res.style)} for listing {hilight(listing.title)}."""
                )
            return res

        self.connect()

        retries = 0
        while retries < self.config.max_retries:
            self.connect()
            assert self.client is not None
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model or self.default_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can confirm if a user's search criteria matches the item he is interested in.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"""{hilight("[AI-Error]", "fail")} {self.config.name} failed to evaluate {hilight(listing.title)}: {e}"""
                    )
                retries += 1
                # try to initiate a connection
                self.client = None
                time.sleep(5)

        # check if the response is yes
        if self.logger:
            self.logger.debug(f"""{hilight("[AI-Response]", "info")} {pretty_repr(response)}""")

        answer = response.choices[0].message.content or ""
        res = self._parse_ai_response(answer, item_config)
        res.to_cache(listing, item_config, marketplace_config)
        counter.increment(CounterItem.NEW_AI_QUERY, item_config.name)
        return res


class DeepSeekBackend(OpenAIBackend):
    default_model = "deepseek-chat"
    base_url = "https://api.deepseek.com"

    @classmethod
    def get_config(cls: Type["DeepSeekBackend"], **kwargs: Any) -> DeekSeekConfig:
        return DeekSeekConfig(**kwargs)


class OllamaBackend(OpenAIBackend):
    default_model = "deepseek-r1:14b"

    @classmethod
    def get_config(cls: Type["OllamaBackend"], **kwargs: Any) -> OllamaConfig:
        return OllamaConfig(**kwargs)


# Provider mapping system for LangChain integration
def _create_openai_model(config: AIConfig) -> BaseChatModel:
    """Create a ChatOpenAI model instance from AIConfig."""
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required")

    return ChatOpenAI(
        api_key=SecretStr(api_key),
        model=config.model or "gpt-4o",
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        default_headers={
            "X-Title": "AI Marketplace Monitor",
            "HTTP-Referer": "https://github.com/BoPeng/ai-marketplace-monitor",
        },
    )


def _create_openrouter_model(config: AIConfig) -> BaseChatModel:
    """Create a ChatOpenAI model instance configured for OpenRouter."""
    api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key is required")
    base_url = config.base_url or "https://openrouter.ai/api/v1"

    return ChatOpenAI(
        api_key=SecretStr(api_key),
        model=config.model or "gpt-4o",
        base_url=base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        default_headers={
            "X-Title": "AI Marketplace Monitor",
            "HTTP-Referer": "https://github.com/BoPeng/ai-marketplace-monitor",
        },
    )


def _create_deepseek_model(config: AIConfig) -> BaseChatModel:
    """Create a ChatDeepSeek model instance from AIConfig."""
    api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key is required")

    return ChatDeepSeek(
        api_key=SecretStr(api_key),
        model=config.model or "deepseek-chat",
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _create_ollama_model(config: AIConfig) -> BaseChatModel:
    """Create a ChatOllama model instance from AIConfig."""
    # Prepare client kwargs with timeout if specified
    client_kwargs = {}
    if config.timeout is not None:
        client_kwargs["timeout"] = config.timeout
    return ChatOllama(
        model=config.model or "deepseek-r1:14b",
        base_url=config.base_url or "http://localhost:11434",
        num_ctx=4096,  # Default context length
        client_kwargs=client_kwargs if client_kwargs else None,
    )


provider_map = {
    "openai": _create_openai_model,
    "deepseek": _create_deepseek_model,
    "ollama": _create_ollama_model,
    "openrouter": _create_openrouter_model,
}


class LangChainBackend(AIBackend[AIConfig]):
    """Unified LangChain backend for AI providers using the provider mapping system.

    This class is thread-safe. Multiple threads can safely call evaluate() and connect()
    methods concurrently. Internal model state is protected by a reentrant lock.
    """

    def __init__(self, config: AIConfig, logger: Logger | None = None) -> None:
        super().__init__(config, logger)
        self._chat_model: BaseChatModel | None = None
        self._model_lock = threading.RLock()  # Reentrant lock for thread safety
        self._prompt_template: ChatPromptTemplate | None = None

    @classmethod
    def get_config(cls: Type["LangChainBackend"], **kwargs: Any) -> AIConfig:
        """Get configuration for the LangChain backend."""
        config = AIConfig(**kwargs)
        cls._validate_config_compatibility(config)
        return config

    @staticmethod
    def _validate_config_compatibility(config: AIConfig) -> None:
        """Validate configuration for compatibility and completeness.

        Ensures the configuration meets all requirements for LangChain backend operation,
        preserving existing validation patterns and error messages.
        """
        # Validate provider is supported
        if not config.provider:
            raise ValueError("AIConfig must have a provider specified")

        provider_key = config.provider.lower()
        if provider_key not in provider_map:
            supported_providers = ", ".join(provider_map.keys())
            raise ValueError(
                f"Unsupported provider '{config.provider}'. "
                f"Supported providers: {supported_providers}"
            )

        # Provider-specific validation
        if provider_key in ("openai", "openrouter"):
            env_key = "OPENAI_API_KEY" if provider_key == "openai" else "OPENROUTER_API_KEY"
            if not config.api_key and not os.getenv(env_key):
                raise ValueError(f"{config.provider} requires an API key")
            if config.model and not isinstance(config.model, str):
                raise ValueError(f"{config.provider} model must be a string")

        elif provider_key == "deepseek":
            if not config.api_key and not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("DeepSeek requires an API key")
            if config.base_url and config.base_url != "https://api.deepseek.com":
                # Note: Config validation warnings would be handled by the logger passed to the backend instance
                pass

        elif provider_key == "ollama":
            if not config.base_url:
                # Use default Ollama URL if not specified
                config.base_url = "http://localhost:11434"
            if not config.model:
                raise ValueError("Ollama requires a model to be specified")

        # Validate common configuration parameters
        if config.max_retries is not None and (
            not isinstance(config.max_retries, int) or config.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")

        if config.timeout is not None and (
            not isinstance(config.timeout, int) or config.timeout <= 0
        ):
            raise ValueError("timeout must be a positive integer")

    def _validate_thread_safety(self) -> None:
        """Validate thread safety by checking that model access is properly synchronized."""
        # This method ensures thread-safe access patterns are maintained
        if not hasattr(self, "_model_lock"):
            raise RuntimeError("LangChainBackend missing proper thread synchronization")

        # Check if it's the right type of lock
        lock = getattr(self, "_model_lock", None)
        if not hasattr(lock, "acquire") or not hasattr(lock, "release"):
            raise RuntimeError("LangChainBackend missing proper thread synchronization")

        # Verify that the lock is functional
        try:
            acquired = self._model_lock.acquire(blocking=False)
            if acquired:
                self._model_lock.release()
            else:
                raise RuntimeError("Thread synchronization lock is in an invalid state")
        except Exception as e:
            raise RuntimeError("Thread synchronization lock is in an invalid state") from e

    def _map_langchain_exception(self, e: Exception, context: str = "") -> Exception:
        """Map LangChain exceptions to existing error patterns for backward compatibility."""
        error_msg = str(e)
        context_prefix = f"{context}: " if context else ""

        # Map common LangChain exceptions to existing patterns
        if isinstance(e, ImportError):
            return RuntimeError(
                f"{context_prefix}Provider dependencies not installed: {error_msg}. "
                "Install the required LangChain packages."
            )

        if isinstance(e, (ValueError, TypeError)) and any(
            pattern in error_msg.lower()
            for pattern in ["api_key", "authentication", "unauthorized"]
        ):
            return ValueError(
                f"{context_prefix}Authentication error: {error_msg}. Check API key configuration."
            )

        if isinstance(e, (ConnectionError, TimeoutError)) or "timeout" in error_msg.lower():
            return RuntimeError(
                f"{context_prefix}Connection failed: {error_msg}. "
                "Check network connectivity and service availability."
            )

        if isinstance(e, KeyError) and "model" in error_msg.lower():
            return ValueError(
                f"{context_prefix}Invalid model configuration: {error_msg}. "
                "Check model name and availability."
            )

        # For unknown exceptions, wrap in RuntimeError with context
        return RuntimeError(f"{context_prefix}Unexpected error: {error_msg}")

    def _validate_mixed_configuration(self, config: AIConfig) -> List[str]:
        """Handle mixed old/new configuration scenarios gracefully.

        Provides clear guidance when users have configurations that mix
        legacy and new patterns, ensuring smooth migration experience.
        """
        warnings = []

        # Check for potential legacy configuration patterns
        if hasattr(config, "service_provider") and config.provider:
            warnings.append(
                "Both 'service_provider' (legacy) and 'provider' (new) are specified. "
                "Using 'provider' value. Consider removing 'service_provider' from your config."
            )

        # Check for DeepSeek API key migration
        if config.provider and config.provider.lower() == "deepseek":
            if config.api_key and os.getenv("DEEPSEEK_API_KEY"):
                warnings.append(
                    "Both config api_key and DEEPSEEK_API_KEY environment variable are set. "
                    "Using environment variable for better security. "
                    "Consider removing 'api_key' from config and using environment variable only."
                )
            elif config.api_key and not os.getenv("DEEPSEEK_API_KEY"):
                warnings.append(
                    "DeepSeek API key in config file. For better security, consider moving to "
                    "DEEPSEEK_API_KEY environment variable and removing from config file."
                )

        # Check for OpenAI API key migration
        if config.provider and config.provider.lower() == "openai":
            if config.api_key and os.getenv("OPENAI_API_KEY"):
                warnings.append(
                    "Both config api_key and OPENAI_API_KEY environment variable are set. "
                    "Using environment variable for better security. "
                    "Consider removing 'api_key' from config and using environment variable only."
                )
            elif config.api_key and not os.getenv("OPENAI_API_KEY"):
                warnings.append(
                    "OpenAI API key in config file. For better security, consider moving to "
                    "OPENAI_API_KEY environment variable and removing from config file."
                )

        # Check for OpenRouter API key migration
        if config.provider and config.provider.lower() == "openrouter":
            if config.api_key and os.getenv("OPENROUTER_API_KEY"):
                warnings.append(
                    "Both config api_key and OPENROUTER_API_KEY environment variable are set. "
                    "Using environment variable for better security. "
                    "Consider removing 'api_key' from config and using environment variable only."
                )
            elif config.api_key and not os.getenv("OPENROUTER_API_KEY"):
                warnings.append(
                    "OpenRouter API key in config file. For better security, consider moving to "
                    "OPENROUTER_API_KEY environment variable and removing from config file."
                )

        # Check for deprecated base URLs or common misconfigurations
        if config.provider and config.provider.lower() == "ollama":
            if config.base_url and config.base_url != "http://localhost:11434":
                warnings.append(
                    f"Custom Ollama base URL detected: {config.base_url}. "
                    "Ensure this URL is correct for your Ollama installation."
                )
            if not config.model:
                warnings.append(
                    "No model specified for Ollama. This may cause connection issues. "
                    "Consider specifying a model like 'llama2' or 'codellama'."
                )

        # General configuration best practices
        if config.api_key and len(config.api_key) < 10:
            warnings.append(
                "API key appears unusually short. Please verify it's correct and complete."
            )

        # Log warnings if logger is available
        if warnings and self.logger:
            for warning in warnings:
                self.logger.warning(f"Configuration migration: {warning}")

        return warnings  # Return for testing purposes

    def _suggest_configuration_improvements(self, config: AIConfig) -> str:
        """Generate configuration improvement suggestions for users.

        Provides actionable recommendations for optimizing configuration
        based on current settings and best practices.
        """
        suggestions = []

        # Environment variable recommendations by provider
        if config.provider and config.api_key:
            provider_lower = config.provider.lower()
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }

            if provider_lower in env_var_map:
                env_var = env_var_map[provider_lower]
                suggestions.append(
                    f"Move API key to environment variable {env_var}:\n"
                    f"  export {env_var}='your_api_key_here'\n"
                    f"  # Then remove 'api_key' from your config file"
                )

        # Model recommendations
        if config.provider:
            provider_lower = config.provider.lower()
            if provider_lower == "openai" and not config.model:
                suggestions.append(
                    "Consider specifying a model for better performance:\n"
                    "  model = 'gpt-3.5-turbo'  # Fast and cost-effective\n"
                    "  # or model = 'gpt-4'     # More capable but slower/costlier"
                )
            elif provider_lower == "ollama" and not config.model:
                suggestions.append(
                    "Ollama requires a model specification:\n"
                    "  model = 'llama2'      # General purpose\n"
                    "  # or model = 'codellama'  # Better for code understanding"
                )
            elif provider_lower == "deepseek" and not config.model:
                suggestions.append(
                    "Consider specifying a DeepSeek model:\n"
                    "  model = 'deepseek-coder'  # Optimized for code tasks\n"
                    "  # or model = 'deepseek-chat'  # General conversation"
                )

        # Timeout and retry recommendations
        if config.timeout and config.timeout < 30:
            suggestions.append(
                "Consider increasing timeout for AI marketplace analysis:\n"
                "  timeout = 60  # Allows more time for complex evaluations"
            )

        if not config.max_retries or config.max_retries < 2:
            suggestions.append(
                "Consider enabling retries for better reliability:\n"
                "  max_retries = 3  # Retry failed requests up to 3 times"
            )

        if suggestions:
            header = f"\n--- Configuration Suggestions for {config.name} ---"
            footer = "--- End Suggestions ---\n"
            return header + "\n\n" + "\n\n".join(suggestions) + "\n" + footer

        return ""

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a structured ChatPromptTemplate for marketplace evaluation."""
        if self._prompt_template is None:
            self._prompt_template = ChatPromptTemplate.from_messages(
                [("system", MARKETPLACE_EVALUATION_SYSTEM_MESSAGE), ("user", "{prompt}")]
            )
        return self._prompt_template

    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent prompt injection attacks."""
        if not text:
            return text

        # HTML escape to prevent script injection
        sanitized = html.escape(text, quote=True)

        # Filter only clear prompt injection attempts (more conservative approach)
        # Focus on patterns that are unlikely to appear in legitimate marketplace listings
        dangerous_patterns = [
            # Only filter exact system tokens that are clearly injection attempts
            (r"\[SYSTEM\]|\<\|system\|\>", "[USER INPUT]"),
            (r"\[ASSISTANT\]|\<\|assistant\|\>", "[USER INPUT]"),
            (r"\[USER\]|\<\|user\|\>", "[USER INPUT]"),
            # Limit newlines to prevent prompt structure breaking
            (r"\n{5,}", "\n\n"),
        ]

        for pattern, replacement in dangerous_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using a more accurate heuristic."""
        if not text:
            return 0

        # More sophisticated token estimation
        # Count words, considering punctuation and special tokens
        words = text.split()
        word_tokens = len(words)

        # Add tokens for punctuation and special characters
        punctuation_tokens = len([c for c in text if c in ".,!?;:()[]{}\"'-/\\"])

        # Add tokens for numbers (numbers often tokenize as single tokens)
        number_tokens = len(re.findall(r"\d+", text))

        # Estimate subword tokens (many words split into multiple tokens)
        estimated_subword_tokens = sum(max(1, len(word) // 4) for word in words)

        # Use the higher estimate between word count and subword estimation
        estimated_tokens = (
            max(word_tokens, estimated_subword_tokens) + punctuation_tokens + number_tokens
        )

        return estimated_tokens

    def _validate_prompt(self, prompt: str) -> None:
        """Validate prompt length and content."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        estimated_tokens = self._estimate_tokens(prompt)
        if estimated_tokens > 8000:  # Conservative limit for most models
            if self.logger:
                self.logger.warning(
                    f"Prompt is very long ({estimated_tokens} estimated tokens). "
                    "Consider shortening for better performance."
                )

    def get_structured_prompt(
        self,
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> ChatPromptTemplate:
        """Create a structured ChatPromptTemplate with the evaluation data."""
        prompt_content = self.get_prompt(listing, item_config, marketplace_config)
        self._validate_prompt(prompt_content)

        template = self._create_prompt_template()
        return template

    def _get_model(self, config: AIConfig) -> BaseChatModel:
        """Retrieve the appropriate LangChain chat model instance based on provider mapping."""
        if not config.provider:
            raise ValueError("AIConfig must have a provider specified")

        provider_key = config.provider.lower()
        if provider_key not in provider_map:
            supported_providers = ", ".join(provider_map.keys())
            raise ValueError(
                f"Unsupported provider '{config.provider}'. "
                f"Supported providers: {supported_providers}"
            )

        try:
            provider_factory = provider_map[provider_key]
            return provider_factory(config)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(
                f"Provider '{config.provider}' configuration error: {e}. "
                f"Check API key, model name, and other required settings."
            ) from e
        except ImportError as e:
            raise RuntimeError(
                f"Provider '{config.provider}' dependencies not installed: {e}. "
                f"Install the required LangChain packages."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create model for provider '{config.provider}': {e}"
            ) from e

    def connect(self) -> None:
        """Establish connection and initialize the chat model."""
        # Validate thread safety before proceeding
        self._validate_thread_safety()

        # Handle mixed configuration scenarios
        self._validate_mixed_configuration(self.config)

        # Show configuration suggestions if enabled
        if os.getenv("AI_MARKETPLACE_MONITOR_SHOW_CONFIG_TIPS", "false").lower() == "true":
            suggestions = self._suggest_configuration_improvements(self.config)
            if suggestions and self.logger:
                self.logger.info(suggestions)

        with self._model_lock:
            if self._chat_model is None:
                try:
                    self._chat_model = self._get_model(self.config)
                    if self.logger:
                        self.logger.info(
                            f"""{hilight("[AI]", "name")} {self.config.name} connected."""
                        )
                except Exception as e:
                    # Map LangChain exceptions to existing error patterns
                    mapped_exception = self._map_langchain_exception(
                        e, f"Connection failed for {self.config.name}"
                    )
                    if self.logger:
                        self.logger.error(
                            f"""{hilight("[AI-Error]", "fail")} Failed to connect {self.config.name}: {mapped_exception}"""
                        )
                    raise mapped_exception from e

    def _extract_response_content(self, response: Any) -> str:
        """Extract content from LangChain response with proper type checking and fallbacks."""
        # Handle different LangChain response types
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and len(content) > 0:
                # Some models return content as a list of message parts
                return str(content[0]) if hasattr(content[0], "__str__") else str(content)

        # Try other common attributes
        if hasattr(response, "text"):
            return str(response.text)

        if hasattr(response, "message") and hasattr(response.message, "content"):
            return str(response.message.content)

        # Log unknown response type for debugging
        if self.logger:
            self.logger.warning(
                f"Unknown response type {type(response)}, falling back to string conversion"
            )

        return str(response)

    def evaluate(
        self,
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> AIResponse:
        """Evaluate a listing using the LangChain model."""
        counter.increment(CounterItem.AI_QUERY, item_config.name)
        prompt = self.get_prompt(listing, item_config, marketplace_config)

        # Check cache first
        res: AIResponse | None = AIResponse.from_cache(listing, item_config, marketplace_config)
        if res is not None:
            if self.logger:
                self.logger.debug(
                    f"""{hilight("[AI]", res.style)} {self.config.name} previously concluded {hilight(f"{res.conclusion} ({res.score}): {res.comment}", res.style)} for listing {hilight(listing.title)}."""
                )
            return res

        self.connect()

        retries = 0
        response = None
        while retries < self.config.max_retries:
            try:
                # Thread-safe access to model
                with self._model_lock:
                    current_model = self._chat_model
                    assert current_model is not None

                # Create structured prompt template and format it
                prompt_template = self._create_prompt_template()
                formatted_messages = prompt_template.format_messages(prompt=prompt)

                # Use LangChain's invoke method with formatted messages
                response = current_model.invoke(formatted_messages)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"""{hilight("[AI-Error]", "fail")} {self.config.name} failed to evaluate {hilight(listing.title)}: {e}"""
                    )
                retries += 1
                # Thread-safe reset of connection on error
                with self._model_lock:
                    self._chat_model = None
                time.sleep(5)
                if retries < self.config.max_retries:
                    try:
                        self.connect()
                    except Exception as connect_error:
                        if self.logger:
                            self.logger.error(
                                f"""{hilight("[AI-Error]", "fail")} {self.config.name} failed to reconnect (attempt {retries + 1}): {connect_error}"""
                            )
                        # Continue to next retry iteration

        # Check if we got a response
        if response is None:
            counter.increment(CounterItem.FAILED_AI_QUERY, item_config.name)
            raise ValueError(
                f"Failed to get response from {self.config.name} after {self.config.max_retries} retries"
            )

        # Parse the response content
        if self.logger:
            self.logger.debug(f"""{hilight("[AI-Response]", "info")} {pretty_repr(response)}""")

        # Extract content using proper type checking with fallbacks
        answer = self._extract_response_content(response)
        res = self._parse_ai_response(answer, item_config)
        res.to_cache(listing, item_config, marketplace_config)
        counter.increment(CounterItem.NEW_AI_QUERY, item_config.name)
        return res
