# AIResponse Cache Integration Tests

This document describes the comprehensive integration test suite for AIResponse caching behavior validation, implemented for **Task 9.3**.

## Test Coverage

### `/tests/test_ai_cache_integration.py`

The integration test suite provides comprehensive validation of AIResponse caching functionality with the LangChain backend integration:

#### 1. Cache Hit/Miss Scenarios
- **`test_cache_hit_miss_basic_scenario`**: Validates basic cache operations (miss → cache → hit)
- Tests cache key generation based on listing, item_config, and marketplace_config hashes
- Verifies complete metadata preservation across cache operations

#### 2. Cache Invalidation
- **`test_cache_invalidation_on_listing_change`**: Ensures cache miss when listing content changes
- **`test_cache_invalidation_on_config_change`**: Validates cache invalidation on config modifications
- Tests that cache keys properly reflect content changes via hash differences

#### 3. Serialization Consistency
- **`test_serialization_consistency_across_backends`**: Tests all AI backends (OpenAI, Anthropic, DeepSeek, OpenRouter)
- Validates that complex metadata structures serialize/deserialize correctly
- Ensures provider-specific metadata formats are preserved

#### 4. Performance Validation
- **`test_cache_performance_improvement_validation`**: Measures cache read/write performance
- Validates performance with large datasets (50+ responses)
- Ensures cache operations complete within acceptable time limits

#### 5. Migration Compatibility
- **`test_legacy_cache_migration_compatibility`**: Tests backward compatibility
- Handles legacy cached data missing new metadata fields
- Supports multiple legacy data formats (minimal, partial metadata, None values)

#### 6. LangChain Integration
- **`test_langchain_backend_cache_integration`**: Full LangChain workflow integration
- Tests complete evaluate() → adapt_langchain_response() → cache → retrieve cycle
- Mocks LangChain responses with comprehensive metadata

#### 7. Edge Cases & Robustness
- **`test_concurrent_cache_access_thread_safety`**: Multi-threaded cache access validation
- **`test_cache_behavior_with_large_responses`**: Large metadata handling (>2MB responses)
- **`test_cache_edge_cases_and_error_handling`**: Corrupted data handling
- **`test_adapt_langchain_response_cache_integration`**: Provider-specific adaptation testing

## Key Features Tested

### Metadata Preservation
- Token usage tracking (prompt_tokens, completion_tokens, total_tokens)
- Provider-specific usage_metadata (reasoning tokens, cache hits, etc.)
- Response metadata (model info, system fingerprints, finish reasons)

### Cache Key Strategy
```python
cache_key = (
    CacheType.AI_INQUIRY.value,  # "ai-inquiries"
    item_config.hash,
    marketplace_config.hash,
    listing.hash,
)
```

### Migration Strategy
- Graceful handling of legacy data without metadata fields
- Automatic population of missing fields with empty dicts
- Error recovery for corrupted cache entries

### Performance Benchmarks
- Cache writes: < 100ms
- Cache reads: < 10ms average
- Large dataset operations: < 50ms per item
- Concurrent access: Thread-safe with no data corruption

## Usage

### Running the Tests
```bash
# Run all integration tests
uv run pytest tests/test_ai_cache_integration.py -v

# Run specific test category
uv run pytest tests/test_ai_cache_integration.py::TestAIResponseCacheIntegration::test_cache_hit_miss_basic_scenario -v

# Run with performance timing
uv run pytest tests/test_ai_cache_integration.py -v -s
```

### Test Fixtures Used
- `temp_cache`: Isolated diskcache instance for testing
- `listing`: Sample Facebook Marketplace listing
- `item_config`: Facebook item configuration
- `marketplace_config`: Facebook marketplace configuration

## Integration with Existing Codebase

The tests integrate with:
- `ai_marketplace_monitor.ai.AIResponse`: Core response class
- `ai_marketplace_monitor.ai.LangChainBackend`: AI backend implementation
- `ai_marketplace_monitor.utils.CacheType`: Cache type enumeration
- Existing test fixtures from `conftest.py`

## Validation Results

✅ All 11 integration tests pass
✅ No performance regressions
✅ Backward compatibility maintained
✅ Thread safety verified
✅ Error handling robust

## Notes

- Tests are designed to run in isolation to avoid cache pollution
- Comprehensive mocking prevents external API calls during testing
- Performance assertions are conservative to account for CI/CD environments
- Edge case testing ensures production stability
