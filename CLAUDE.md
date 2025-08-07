# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Marketplace Monitor is a Python tool that monitors Facebook Marketplace using AI to help find deals. It uses Playwright for web scraping, supports multiple AI backends (OpenAI, DeepSeek, Ollama), and provides various notification methods (PushBullet, Telegram, Email, etc.).

## Development Commands

**Package Management & Environment:**
- Use `uv` for all package management operations
- `uv run <command>` to run commands in the virtual environment

**Testing:**
- `uv run invoke tests` - Run all tests with coverage
- `uv run pytest tests/` - Run tests directly
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run pytest -xvs` - Run tests with verbose output and stop on first failure

**Code Quality:**
- `uv run invoke lint` - Run all linting (ruff + format check)
- `uv run invoke ruff` - Run ruff linter
- `uv run invoke format` - Format code with black and isort
- `uv run invoke mypy` - Run type checking

**Coverage:**
- `uv run invoke coverage` - Generate coverage report
- `uv run invoke coverage --fmt html --open-browser` - Generate HTML coverage report and open in browser

**Documentation:**
- `uv run invoke docs` - Build documentation
- `uv run invoke docs --serve --open-browser` - Build docs with live reload

**Security:**
- `uv run invoke security` - Run security checks with pip-audit

**Pre-commit:**
- `uv run invoke install-hooks` - Install pre-commit hooks
- `uv run invoke hooks` - Run pre-commit hooks manually

## Code Architecture

**Main Entry Points:**
- `src/ai_marketplace_monitor/cli.py` - Command-line interface using Typer
- `src/ai_marketplace_monitor/monitor.py` - Core monitoring logic and MarketplaceMonitor class

**Key Components:**
- `config.py` - Configuration management with TOML support
- `marketplace.py` - Marketplace abstraction layer
- `facebook.py` - Facebook Marketplace implementation
- `listing.py` - Listing data structures and parsing
- `ai.py` - AI backend integrations (OpenAI, DeepSeek, Ollama)
- `notification.py` - Notification system orchestration
- `user.py` - User management and notification preferences

**Notification Modules:**
- `email_notify.py` - Email notifications with HTML templates
- `telegram.py` - Telegram bot notifications
- `pushbullet.py` - PushBullet notifications
- `pushover.py` - PushOver notifications
- `ntfy.py` - Ntfy notifications

**Utility Modules:**
- `utils.py` - Common utilities, caching, keyboard monitoring
- `region.py` - Geographic region definitions

## Configuration System

The project uses TOML configuration files with sections for:
- `[ai.*]` - AI backend configurations
- `[marketplace.*]` - Marketplace settings
- `[item.*]` - Search item configurations
- `[user.*]` - User and notification settings
- `[notification.*]` - Shared notification configurations

Default config location: `~/.ai-marketplace-monitor/config.toml`

## Testing Strategy

- Tests use pytest with asyncio support
- Playwright tests for browser automation
- Mock external services (AI APIs, notification services)
- Coverage target: 100% (with pragmas for exclusions)
- Test files mirror the source structure in `tests/`

## Important Implementation Notes

**Web Scraping:**
- Uses Playwright for browser automation
- Supports headless and headed modes
- Handles Facebook login and CAPTCHA challenges
- Parses different Facebook Marketplace layouts

**AI Integration:**
- Supports multiple AI providers via OpenAI-compatible APIs
- Uses structured prompts for listing evaluation
- Implements rating system (1-5) for listings
- Caches AI responses to reduce API costs

**Caching System:**
- Uses diskcache for persistent storage
- Caches listing details, AI inquiries, user notifications
- Cache types: listing-details, ai-inquiries, user-notification, counters

**Notification System:**
- Supports multiple notification methods per user
- HTML email templates with embedded images
- Rate limiting and duplicate prevention
- Configurable notification levels based on AI ratings

## Development Guidelines

**Code Style:**
- Line length: 99 characters (configured in ruff/black)
- Use Google-style docstrings
- Type hints required for all functions
- Ruff for linting with extensive rule set

**Security Considerations:**
- Never commit API keys or tokens
- Use environment variable substitution in configs
- Validate all external inputs
- Sanitize data before AI processing

**Performance:**
- Cache expensive operations (web scraping, AI calls)
- Use async patterns where appropriate
- Monitor memory usage for long-running processes
- Implement proper cleanup for browser resources

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
