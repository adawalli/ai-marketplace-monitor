# LangChain Migration PRD
**AI Marketplace Monitor: OpenAI SDK to LangChain Migration**

## Overview
Migrate AI Marketplace Monitor from direct OpenAI SDK usage to LangChain-based architecture while maintaining 100% backward compatibility. This migration will unify all AI providers (OpenAI, DeepSeek, Ollama, OpenRouter) under a single LangChain interface, enabling simplified provider management and future enhanced features while preserving existing TOML configuration syntax.

The core problem this solves is the current fragmented AI backend system where each provider requires separate SDK integration and maintenance. LangChain provides a unified interface that simplifies provider management through consistent chat model abstractions and creates a foundation for future AI workflow enhancements.

## Core Features

### 1. Unified LangChain Backend Architecture
**What it does**: Replace all existing AI backends (OpenAIBackend, DeepSeekBackend, OllamaBackend) with a single LangChainBackend that internally routes to appropriate LangChain chat models.

**Why it's important**: Eliminates code duplication, provides consistent error handling and retry logic across all providers, and creates a single point of maintenance for AI integrations.

**How it works**:
- Single `LangChainBackend` class replaces three separate backend classes
- Provider-specific configuration automatically maps to appropriate LangChain chat models
- All existing TOML configurations work unchanged through internal routing

### 2. OpenRouter Provider Integration
**What it does**: Add OpenRouter as a new provider type alongside existing OpenAI, DeepSeek, and Ollama options.

**Why it's important**: Provides access to 200+ AI models through a single API, enabling cost optimization and model experimentation without changing infrastructure.

**How it works**:
```toml
[ai.claude_sonnet]
provider = "openrouter"
api_key = "${OPENROUTER_API_KEY}"
model = "anthropic/claude-3-sonnet"
```

### 3. AI Cost Tracking System (Phase 2)
**What it does**: Track and report costs for all AI provider interactions with per-model, per-item, and per-timeframe breakdowns.

**Why it's important**: Users can optimize costs by understanding which models provide best value for their use cases and set budgets to prevent unexpected expenses.

**How it works**:
- Capture token usage and model pricing for each evaluation
- Store cost data in existing cache system with new CacheType.AI_COSTS
- Provide cost summaries in CLI output and potential future UI
- Support cost budgets and alerts

**Implementation Note**: This feature will be implemented in Phase 2 after core migration is validated.

### 4. Optional LangSmith Integration
**What it does**: Enable advanced observability, tracing, and cost monitoring through LangSmith platform.

**Why it's important**: Provides enterprise-grade monitoring, debugging capabilities, and automated cost tracking with minimal setup effort.

**How it works**:
- Simple environment variable configuration (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`)
- Automatic tracing of all LangChain LLM calls without code changes
- Real-time cost tracking and usage dashboards
- Advanced features like prompt versioning and performance monitoring
- **Optional**: Users can enable by setting environment variables, no impact if not configured

### 5. Backward Configuration Compatibility
**What it does**: Ensure all existing `[ai.name]` TOML configurations continue working without any changes required from users.

**Why it's important**: Zero-disruption migration allows users to benefit from improvements without configuration work or learning new syntax.

**How it works**:
- Internal provider mapping converts existing configs to LangChain models
- Preserve all existing config validation and error handling
- Maintain identical API surface for Config.get_ai_config() method

## Technical Architecture

### System Components

#### Core Architecture Changes
```python
# Before: Multiple backend classes
class OpenAIBackend(AIBackend): ...
class DeepSeekBackend(OpenAIBackend): ...
class OllamaBackend(OpenAIBackend): ...

# After: Single unified backend
class LangChainBackend(AIBackend):
    def _get_model(self, config: AIConfig) -> BaseChatModel:
        provider_map = {
            "openai": lambda c: ChatOpenAI(api_key=c.api_key, model=c.model),
            "deepseek": lambda c: ChatDeepSeek(model=c.model),  # Uses DEEPSEEK_API_KEY env var
            "ollama": lambda c: ChatOllama(base_url=c.base_url, model=c.model),
            "openrouter": lambda c: ChatOpenAI(
                api_key=c.api_key,
                model=c.model,
                base_url="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://ai-marketplace-monitor",
                               "X-Title": "AI Marketplace Monitor"}
            ),
        }
        return provider_map[config.provider](config)
```

#### Data Models
- **CostTracker**: New class for tracking AI usage costs per provider/model
- **AIConfig Extensions**: Add optional cost_tracking and budget fields
- **AIResponse Extensions**: Include cost information in response objects

#### APIs and Integrations
- **LangChain Integration**: ChatOpenAI, ChatDeepSeek, ChatOllama (OpenRouter uses ChatOpenAI with custom base_url)
- **Cost APIs**: Integration with provider pricing APIs for real-time cost calculation (Phase 2)
- **Cache Extensions**: New cache types for cost data and model metadata (Phase 2)

#### Infrastructure Requirements
- **Dependencies**: Add LangChain packages with compatible version constraints:
  - `langchain-core>=0.3.5,<0.4.0`
  - `langchain-openai>=0.3.5,<0.4.0`
  - `langchain-community>=0.0.10,<0.1.0`
  - `langchain-deepseek` (latest compatible with core 0.3.x)
  - `langsmith` (optional, for enhanced tracing and cost monitoring)
- **Configuration**: Extend existing TOML structure with new provider types
- **Logging**: Enhanced logging for LangChain operations and cost tracking
- **Python Compatibility**: Maintain existing Python 3.10+ requirements

## Development Roadmap

### Phase 1: Core LangChain Migration
**Goal**: Replace existing AI backends with unified LangChain interface while maintaining 100% backward compatibility.

**Requirements**:
- Replace all existing AI backends with unified LangChainBackend
- Implement provider mapping for OpenAI, DeepSeek, Ollama, and OpenRouter
- Create response adapter layer to maintain AIResponse compatibility
- Implement LangChain exception mapping to existing error patterns
- Add comprehensive unit tests for new backend system
- Create backward compatibility test suite using existing configurations
- Test migration with existing cached data
- Update internal documentation

**Acceptance Criteria**:
- All existing TOML configurations work without modification
- AIResponse format and caching remain identical (verified with existing cache)
- Zero breaking changes to public API
- All existing error handling patterns preserved
- DeepSeek configurations migrate to environment variable approach

### Phase 2: Cost Tracking & Enhanced Features
**Goal**: Add cost tracking system and optimization features after core migration is validated.

**Requirements**:
- Implement CostTracker class with per-provider cost calculation
- Add cost information to AIResponse objects
- Extend cache system with CacheType.AI_COSTS
- Create cost reporting utilities for CLI output
- Add optional budget configuration and alerts
- **LangSmith Integration**: Optional enhanced monitoring through environment variables
- Implement A/B testing framework for comparing models on same listings
- Add model performance metrics tracking (response time, success rate)
- Create cost optimization recommendations based on usage patterns

## Logical Dependency Chain

### Foundation Layer (Must Build First)
1. **LangChain Dependencies**: Install and configure langchain packages with proper version constraints
2. **Provider Interface Contract**: Define standardized interface for all providers
3. **Provider Mapping System**: Core provider-to-model mapping functionality
4. **LangChainBackend Base**: Basic backend class with connection handling

### Migration Layer (Core Functionality)
5. **Configuration Compatibility**: Ensure existing configs work with new backend
6. **Provider Integration**: OpenAI, DeepSeek, Ollama, OpenRouter routing through LangChain
7. **Response Adapter Layer**: Convert LangChain responses to AIResponse format
8. **Error Handling Migration**: Map LangChain exceptions to existing error patterns
9. **Caching Integration**: Ensure AIResponse caching works with new backend
10. **Environment Variable Migration**: Handle DeepSeek auth change from config to env var
11. **Backward Compatibility Testing**: Test with real existing configurations and cached data

### Enhancement Layer (Phase 2 Features)
10. **Cost Tracking Infrastructure**: Base cost calculation and storage
11. **LangSmith Integration**: Optional environment variable-based tracing and monitoring
12. **Cost Reporting**: CLI integration and user-facing cost information
13. **Performance Monitoring**: Model response time and reliability tracking
14. **A/B Testing Framework**: Compare models on identical evaluation tasks

### Optimization Layer (Future Improvements)
14. **Smart Model Selection**: Automatic model routing based on cost/performance
15. **Advanced Multi-Agent Workflows**: Future LangGraph integration for complex scenarios

## Risks and Mitigations

### Technical Challenges
**Risk**: LangChain version compatibility issues with existing Python 3.10+ constraints
**Mitigation**: Use tested compatible versions (core/openai 0.3.5+, community 0.0.10+); validate in isolated environment

**Risk**: DeepSeek integration requires environment variable configuration changes
**Mitigation**: Update configuration system to handle environment variable-based auth for DeepSeek provider

**Risk**: Performance regression from additional LangChain abstraction layer
**Mitigation**: Benchmark critical paths during development; LangChain adds minimal overhead for single-model use cases

**Risk**: Breaking changes in provider API compatibility through LangChain updates
**Mitigation**: Pin LangChain versions with tested ranges (`>=0.3.5,<0.4.0`); implement provider-specific integration tests

**Risk**: Existing AIResponse caching may not work with LangChain response objects
**Mitigation**: Create response adapter layer to maintain cache compatibility; test serialization with existing cached data

**Risk**: LangChain exception hierarchy differs from OpenAI SDK
**Mitigation**: Create exception mapping layer to preserve existing error handling patterns

### Migration Complexity
**Risk**: Subtle behavioral differences between direct SDK and LangChain implementations
**Mitigation**: Comprehensive backward compatibility testing with existing configurations and real marketplace data

**Risk**: Configuration edge cases not handled by new unified backend
**Mitigation**: Extensive unit testing of configuration parsing and validation logic

### Future-Proofing
**Risk**: LangChain architecture may limit future multi-agent features
**Mitigation**: LangChain provides solid foundation for current needs; can integrate LangGraph later for advanced multi-agent workflows if needed

**Risk**: Cost tracking accuracy may be impacted by provider pricing changes
**Mitigation**: Implement fallback cost estimation and clear disclaimers about cost approximations; focus on relative cost comparison rather than absolute accuracy

## Appendix

### Research Findings
- LangChain provides built-in support for all target providers through langchain-community
- OpenRouter uses OpenAI-compatible API, requires ChatOpenAI with custom base_url and headers
- DeepSeek integration requires `langchain-deepseek` package and environment variable authentication
- Cost tracking can leverage provider token counting mechanisms available in LangChain (Phase 2)
- LangSmith provides automatic cost tracking and tracing with just environment variables - no code changes needed
- LangChain response objects require adapter layer for existing AIResponse compatibility
- Compatible versions: core/openai 0.3.5+, community 0.0.10+, all using same core version
- LangChain is optimal for simple provider unification vs LangGraph's multi-agent complexity

### Technical Specifications
- **Python Version**: Maintain compatibility with existing project requirements (3.10+)
- **LangChain Versions**:
  - `langchain-core>=0.3.5,<0.4.0`
  - `langchain-openai>=0.3.5,<0.4.0`
  - `langchain-community>=0.0.10,<0.1.0`
  - `langchain-deepseek` (latest compatible)
  - `langsmith` (optional for enhanced monitoring)
- **Provider APIs**: All existing provider endpoints remain unchanged
- **Configuration Format**: TOML structure preserved exactly as current implementation
- **Cache Compatibility**: AIResponse objects maintain same serialization format
- **OpenRouter Integration**: Uses ChatOpenAI with `base_url="https://openrouter.ai/api/v1"` and required headers
- **DeepSeek Integration**: Uses `langchain-deepseek` package with `DEEPSEEK_API_KEY` environment variable
- **LangSmith Integration**: Optional tracing via `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` environment variables
- **Error Handling**: LangChain exceptions mapped to existing error patterns

### Provider Interface Contract
```python
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from typing import Dict, Any

class ProviderInterface(ABC):
    @abstractmethod
    def get_model(self, config: AIConfig) -> BaseChatModel:
        """Return configured LangChain model with standardized parameters"""

    @abstractmethod
    def map_config(self, config: AIConfig) -> Dict[str, Any]:
        """Map AIConfig to provider-specific parameters"""

    @abstractmethod
    def handle_errors(self, error: Exception) -> Exception:
        """Map provider exceptions to existing error patterns"""

    @abstractmethod
    def adapt_response(self, langchain_response) -> AIResponse:
        """Convert LangChain response to existing AIResponse format"""
```

### Migration Testing Strategy
```python
# Backward Compatibility Test Suite
class MigrationTestSuite:
    def test_existing_configs(self):
        """Test all existing TOML configurations work unchanged"""

    def test_cached_data_compatibility(self):
        """Verify existing cached AIResponse objects still work"""

    def test_error_handling_preservation(self):
        """Ensure existing error patterns are maintained"""

    def test_deepseek_env_migration(self):
        """Test DeepSeek config migration to environment variables"""
```

### Future Multi-Agent Possibilities
While not part of this migration, the LangChain foundation enables potential future LangGraph integration for:
- **Consensus Scoring**: Multiple models evaluating high-value items for increased confidence
- **Category Specialists**: Different models optimized for specific item categories
- **Validation Agents**: Secondary models for catching false positives/negatives
- **Price Analysis**: Dedicated agents for market price comparison

These advanced workflows could be implemented later using LangGraph while preserving the LangChain foundation.
