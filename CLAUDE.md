# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

## Essential Commands

### Development Workflow
```bash
# Setup and dependencies
uv sync                           # Install all dependencies
uv run playwright install        # Install browser for marketplace scraping

# Linting and formatting
uv run invoke lint               # Run ruff + format check (pre-commit style)
uv run invoke format             # Format code with isort + black
uv run invoke ruff               # Run ruff linter only

# Testing
uv run invoke tests              # Run full test suite with coverage
uv run pytest tests/test_specific.py::test_function  # Run single test
uv run invoke coverage           # Generate coverage report
uv run invoke coverage --fmt=html --open-browser     # HTML coverage report

# Type checking and security
uv run invoke mypy               # Type checking
uv run invoke security           # Security audit with pip-audit

# Documentation
uv run invoke docs               # Build documentation
uv run invoke docs --serve --open-browser  # Live docs server

# Pre-commit hooks
uv run invoke install-hooks      # Install pre-commit hooks
uv run invoke hooks              # Run all pre-commit hooks manually

# Application execution
ai-marketplace-monitor           # Run the main application
ai-marketplace-monitor --headless  # Run without browser window
ai-marketplace-monitor --config path/to/config.toml  # Custom config
```

### Development with nox (multi-Python testing)
```bash
nox -s tests                     # Run tests across Python 3.10, 3.11, 3.12
nox -s mypy                      # Type checking across versions
nox -s coverage                  # Coverage reporting
nox -s security                  # Security scanning
```

## Architecture Overview

### Core Application Flow
**AI Marketplace Monitor** is a synchronous marketplace monitoring application that uses Playwright for web scraping, AI services for listing evaluation, and multiple notification backends for user alerts.

**Main execution path:**
1. `MarketplaceMonitor` (monitor.py) - Central orchestrator
2. `Config` (config.py) - TOML configuration loading and validation
3. `Marketplace` implementations (facebook.py) - Platform-specific scraping
4. `AIBackend` services (ai.py) - Listing evaluation
5. `NotificationConfig` subclasses - Multi-channel user notifications

### Configuration System
- **TOML-based configuration** with system defaults merged with user configs
- **Hierarchical structure:** `[marketplace]`, `[item]`, `[user]`, `[ai]`, `[notification]` sections
- **Dynamic configuration reloading** - monitors file changes during runtime
- **Type-safe config classes** using dataclasses with validation handlers

### Notification Architecture
**Plugin-based notification system** with inheritance hierarchy:
```
NotificationConfig (base)
├── PushNotificationConfig (push notification base)
│   ├── PushbulletNotificationConfig
│   ├── PushoverNotificationConfig
│   ├── NtfyNotificationConfig
│   └── TelegramNotificationConfig
└── EmailNotificationConfig
```

**Key patterns for adding notification backends:**
- Extend `PushNotificationConfig` for push-style notifications
- Define `required_fields` class variable for validation
- Implement `send_message(title, message, logger)` method
- Use `send_message_with_retry()` from base class for error handling

### Marketplace Integration
**Playwright-based web scraping** with marketplace-specific implementations:
- `FacebookMarketplace` - Currently the only supported platform
- **Browser management** - Persistent browser instance with automatic login
- **Listing extraction** - Platform-specific HTML parsing and data normalization
- **Search configuration** - Multi-city, region, and filtering support

### AI Integration
**Pluggable AI backends** for listing evaluation:
- `OpenAIBackend`, `DeepSeekBackend`, `OllamaBackend`
- **Structured prompting** - Configurable evaluation criteria and rating system
- **Caching system** - Persistent cache for AI responses and listing details
- **Rating-based filtering** - 1-5 scale with configurable notification thresholds

### Async Integration Considerations
**Current codebase is fully synchronous** but designed for future async notification backends:
- Use `asyncio.run()` pattern for integrating async libraries (telegram, discord)
- Extend `PushNotificationConfig` - never modify base notification classes
- Keep existing sync interfaces unchanged to avoid corruption
- Test async integrations using `AsyncMock` in existing sync test suite

## Testing Strategy

### Test Structure
- **Unit tests** in `tests/` directory following `test_*.py` pattern
- **pytest fixtures** in `conftest.py` for common test objects
- **100% coverage requirement** with meaningful tests (not busy work)
- **Mock-based testing** - No real API calls to external services

### Testing Philosophy: Business Logic Over Implementation Details

**Focus on VALUABLE TESTS** that catch real bugs and test behavior:

✅ **KEEP THESE TEST TYPES:**
- Configuration validation (required fields, formats, edge cases)
- Core business logic (rate limiting, calculations, algorithms)
- Success/failure paths with proper error handling
- Data transformation and preservation (message splitting, parsing)
- Edge cases and boundary conditions

❌ **AVOID THESE TEST TYPES (busywork):**
- Testing how many times internal methods are called
- Complex mocking of external library internals (telegram Bot, etc.)
- Integration tests disguised as unit tests
- Tests that verify implementation details rather than behavior
- Overly complex async/sync boundary testing
- Tests requiring extensive setup for minimal value

### Guidelines for Adding New Tests
1. **Ask: "Would this test catch a real bug?"**
2. **Ask: "Is this testing behavior or implementation?"**
3. Keep tests simple, focused, and maintainable
4. Use realistic test data (proper token formats, valid IDs)
5. Always mock external dependencies (asyncio.run, APIs, etc.)

### Async Testing Guidelines
- **NEVER write direct async test functions** (`async def test_...`)
- **ALWAYS mock asyncio.run()** to prevent event loop conflicts
- Use sync tests that mock async internals before asyncio.run() call
- See `tests/test_notification.py` for examples

### Key Testing Patterns
```python
# Configuration testing - Test validation logic
def test_config_validation():
    config = NotificationConfig(required_field=None)
    assert not config._has_required_fields()

# Business logic testing - Test calculations/algorithms
def test_rate_limit_calculation():
    config.last_send_time = time.time() - 0.5
    wait_time = config._get_wait_time()
    assert 0.4 < wait_time <= 0.6

# Success/failure paths - Test behavior with mocking
def test_send_message_success():
    with patch("asyncio.run", return_value=True):
        result = config.send_message("title", "message", None)
        assert result is True

# Algorithm testing - Test data preservation
def test_message_splitting():
    result = config._split_message("long message", 10)
    rejoined = " ".join(result)
    assert rejoined == "long message"
```

### Running Tests
- **Single test:** `uv run pytest tests/test_file.py::test_function`
- **Test category:** `uv run pytest tests/test_notification.py`
- **With coverage:** `uv run invoke tests` (includes xdoctest)
- **Multiple Python versions:** `nox -s tests`

### Test Quality Success Story
The `tests/test_notification.py` file was refactored from **85+ complex integration tests** down to **20 focused unit tests** while maintaining comprehensive coverage. This demonstrates the value of testing business logic over implementation details.

## Key Implementation Patterns

### Configuration Validation
- **Handler methods:** `handle_fieldname()` methods for field-specific validation
- **Required fields:** `required_fields` class variable for dependency validation
- **Type coercion:** Automatic string/list/int conversions in base classes

### Error Handling
- **Retry logic:** Built into notification base classes with exponential backoff
- **Graceful degradation:** Continue operation when optional services fail
- **User feedback:** Rich console output with color-coded status messages

### Caching Strategy
- **diskcache backend** for persistent storage across application restarts
- **Multi-level caching:** Listing details, AI responses, user notifications
- **Cache keys:** Structured keys for easy management and debugging

### Browser Management
- **Persistent browser instance** managed by `MarketplaceMonitor`
- **Automatic login handling** with user credential management
- **Headless mode support** for server deployments
- **Error recovery** from browser crashes or network issues

## Development Guidelines

### Adding New Notification Backends
1. **Extend `PushNotificationConfig`** (never modify base classes)
2. **Define `required_fields`** and validation handlers
3. **Implement `send_message()`** with proper error handling
4. **Add to notification discovery** in config system
5. **Write unit tests** using mocked API calls
6. **For async libraries:** Use `asyncio.run()` pattern internally

### Adding New Marketplaces
1. **Implement `Marketplace` interface** in new module
2. **Add to `supported_marketplaces`** in config.py
3. **Create marketplace-specific config classes**
4. **Implement browser automation** with Playwright
5. **Add listing extraction logic** with error handling
6. **Create test fixtures** with static HTML samples

## Async Test Pattern Memories

### AI Marketplace Monitor - Async Test Pattern
- Project uses asyncio.run() isolation for async notification backends (Telegram, Discord, etc.)
- NEVER write direct async test functions (async def test_...) - they cause event loop conflicts in full test suite
- ALWAYS write sync tests that mock async internals BEFORE asyncio.run() call
- Pattern: Mock the async method with patch.object(), then call asyncio.run() in the test
- This is documented in docs/telegram_support_prd.md and test_notification.py header
- Event loop conflicts manifest as tests passing individually but failing in full suite
- Task 8.3 provides working examples of proper async test patterns

### Test Quality Maintenance
**Successful Test Cleanup Example (Task 8.3):**
- **Before:** 85+ complex integration tests with event loop conflicts
- **After:** 20 focused unit tests covering essential business logic
- **Result:** 100% pass rate, no event loop conflicts, easier maintenance

**Key Lessons:**
- Tests should catch bugs, not verify implementation details
- Simple, focused tests are more valuable than complex integration tests
- Proper mocking prevents external dependencies from breaking tests
- Testing philosophy documentation helps maintain focus over time

**Red Flags for Future Test Reviews:**
- Tests requiring extensive setup for minimal assertions
- Testing "how many times method X was called"
- Complex mocking of external library internals
- Tests that only pass in isolation but fail in full suite
- Integration tests disguised as unit tests
