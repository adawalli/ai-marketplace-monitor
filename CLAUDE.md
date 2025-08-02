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
│   └── NtfyNotificationConfig
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

### Key Testing Patterns
```python
# Configuration testing
def test_config_validation():
    # Test TOML parsing and validation logic

# Notification testing
def test_notification_retry():
    # Test retry logic and error handling

# Marketplace testing
def test_listing_extraction():
    # Test HTML parsing with static fixtures

# AI backend testing
def test_ai_response_parsing():
    # Test AI response processing with mocked responses
```

### Running Tests
- **Single test:** `uv run pytest tests/test_file.py::test_function`
- **Test category:** `uv run pytest tests/test_notification.py`
- **With coverage:** `uv run invoke tests` (includes xdoctest)
- **Multiple Python versions:** `nox -s tests`

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
