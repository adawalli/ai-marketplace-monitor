import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from logging import Logger
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar

from diskcache import Cache  # type: ignore
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from openai import OpenAI  # type: ignore
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

    def get_prompt(
        self: "AIBackend",
        listing: Listing,
        item_config: TItemConfig,
        marketplace_config: TMarketplaceConfig,
    ) -> str:
        prompt = (
            f"""A user wants to buy a {item_config.name} from Facebook Marketplace. """
            f"""Search phrases: "{'" and "'.join(item_config.search_phrases)}", """
        )
        if item_config.description:
            prompt += f"""Description: "{item_config.description}", """
        #
        max_price = item_config.max_price or 0
        min_price = item_config.min_price or 0
        if max_price and min_price:
            prompt += f"""Price range: {min_price} to {max_price}. """
        elif max_price:
            prompt += f"""Max price {max_price}. """
        elif min_price:
            prompt += f"""Min price {min_price}. """
        #
        if item_config.antikeywords:
            prompt += f"""Exclude keywords "{'" and "'.join(item_config.antikeywords)}" in title or description."""
        #
        prompt += (
            f"""\n\nThe user found a listing titled "{listing.title}" in {listing.condition} condition, """
            f"""priced at {listing.price}, located in {listing.location}, """
            f"""posted at {listing.post_url} with description "{listing.description}"\n\n"""
        )
        # prompt
        if item_config.prompt is not None:
            prompt += item_config.prompt
        elif marketplace_config.prompt is not None:
            prompt += marketplace_config.prompt
        else:
            prompt += (
                "Evaluate how well this listing matches the user's criteria. Assess the description, MSRP, model year, "
                "condition, and seller's credibility."
            )
        # extra_prompt
        prompt += "\n"
        if item_config.extra_prompt is not None:
            prompt += f"\n{item_config.extra_prompt.strip()}\n"
        elif marketplace_config.extra_prompt is not None:
            prompt += f"\n{marketplace_config.extra_prompt.strip()}\n"
        # rating_prompt
        if item_config.rating_prompt is not None:
            prompt += f"\n{item_config.rating_prompt.strip()}\n"
        elif marketplace_config.rating_prompt is not None:
            prompt += f"\n{marketplace_config.rating_prompt.strip()}\n"
        else:
            prompt += (
                "\nRate from 1 to 5 based on the following: \n"
                "1 - No match: Missing key details, wrong category/brand, or suspicious activity (e.g., external links).\n"
                "2 - Potential match: Lacks essential info (e.g., condition, brand, or model); needs clarification.\n"
                "3 - Poor match: Some mismatches or missing details; acceptable but not ideal.\n"
                "4 - Good match: Mostly meets criteria with clear, relevant details.\n"
                "5 - Great deal: Fully matches criteria, with excellent condition or price.\n"
                "Conclude with:\n"
                '"Rating <1-5>: <summary>"\n'
                "where <1-5> is the rating and <summary> is a brief recommendation (max 30 words)."
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
            self.client = OpenAI(
                api_key=self.config.api_key,
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

    return ChatOpenAI(
        api_key=api_key,
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
    base_url = config.base_url or "https://openrouter.ai/api/v1"

    return ChatOpenAI(
        api_key=api_key,
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

    return ChatDeepSeek(
        api_key=api_key,
        model=config.model or "deepseek-chat",
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _create_ollama_model(config: AIConfig) -> BaseChatModel:
    """Create a ChatOllama model instance from AIConfig."""
    return ChatOllama(
        model=config.model or "deepseek-r1:14b",
        base_url=config.base_url or "http://localhost:11434",
        timeout=config.timeout,
        num_ctx=4096,  # Default context length
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

    @classmethod
    def get_config(cls: Type["LangChainBackend"], **kwargs: Any) -> AIConfig:
        """Get configuration for the LangChain backend."""
        return AIConfig(**kwargs)

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

        provider_factory = provider_map[provider_key]
        try:
            return provider_factory(config)
        except KeyError as e:
            raise ValueError(
                f"Provider '{config.provider}' configuration is missing required field: {e}"
            ) from e
        except TypeError as e:
            # Handle invalid parameter types or missing required parameters
            raise ValueError(
                f"Provider '{config.provider}' configuration error: {e}. "
                f"Check API key, model name, and other required settings."
            ) from e
        except ImportError as e:
            raise RuntimeError(
                f"Provider '{config.provider}' dependencies not installed: {e}. "
                f"Install the required LangChain packages."
            ) from e
        except ValueError as e:
            # Re-raise ValueError with more context
            raise ValueError(f"Provider '{config.provider}' configuration error: {e}") from e
        except Exception as e:
            # Preserve original exception details for debugging
            error_msg = f"Failed to create model for provider '{config.provider}': {e}"
            if hasattr(e, "__cause__") and e.__cause__:
                error_msg += f" (caused by: {e.__cause__})"
            raise RuntimeError(error_msg) from e

    def connect(self) -> None:
        """Establish connection and initialize the chat model."""
        with self._model_lock:
            if self._chat_model is None:
                try:
                    self._chat_model = self._get_model(self.config)
                    if self.logger:
                        self.logger.info(
                            f"""{hilight("[AI]", "name")} {self.config.name} connected."""
                        )
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            f"""{hilight("[AI-Error]", "fail")} Failed to connect {self.config.name}: {e}"""
                        )
                    raise

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

                # Use LangChain's invoke method (outside the lock to avoid blocking other threads)
                response = current_model.invoke(
                    [
                        (
                            "system",
                            "You are a helpful assistant that can confirm if a user's search criteria matches the item he is interested in.",
                        ),
                        ("user", prompt),
                    ]
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
