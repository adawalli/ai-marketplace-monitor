# Overview

The AI Marketplace Monitor is currently a synchronous Python application that monitors Facebook Marketplace for items matching user criteria. While functional, the project needs to migrate to an async-first architecture to maintain compatibility with modern Python libraries and prepare for future extensibility. The focus is on simplicity and readability rather than performance optimization, as the current application doesn't have performance issues.

# Core Features

## Current Synchronous Architecture

- **Simple Sequential Flow**: Operations run one after another in a predictable, easy-to-debug manner
- **Synchronous I/O**: Network requests use blocking calls with straightforward error handling
- **Mature Libraries**: Currently uses well-established synchronous libraries (requests, sync Playwright)
- **Straightforward Debugging**: Linear execution makes troubleshooting simpler

## Target Async Benefits (Focused on Simplicity)

- **Modern Library Compatibility**: Enable future integration with async-first Python libraries
- **Maintainable Patterns**: Establish clean async/await patterns throughout the codebase
- **Future-Proofing**: Position the project for ecosystem evolution without major rewrites
- **Readable Async Code**: Implement async patterns that are clear and maintainable

# User Experience

## User Personas

- **End Users**: Continue to experience the same CLI interface and functionality with no disruption
- **Developers**: Work with clean, readable async/await patterns that are easy to understand and extend
- **Future Maintainers**: Benefit from modern async patterns that align with Python ecosystem trends

## Key User Flows (Preserved Behavior)

- **Startup**: Application initializes with async foundation but maintains same startup experience
- **Search Execution**: Same search behavior but implemented with async patterns internally
- **AI Processing**: Same AI evaluation flow but using async HTTP clients
- **Notifications**: Same notification delivery but with async implementation
- **Configuration Updates**: Existing config reloading behavior preserved

## UI/UX Considerations

- Maintain existing CLI interface behavior and output formatting
- Preserve keyboard interrupt handling and interactive features
- Keep existing logging and progress reporting patterns

# Technical Architecture

## System Components Migration

### Core Architecture Changes

- **Event Loop Foundation**: Migrate from sync main loop to asyncio event loop
- **Async Task Management**: Replace `schedule` library with async task scheduling
- **Concurrent Execution**: Transform sequential operations into concurrent async tasks

### Component-Level Async Migration

1. **CLI Layer (`cli.py`)**

   - Wrap async main function with `asyncio.run()`
   - Maintain synchronous CLI interface while enabling async internals

2. **Monitor Core (`monitor.py`)**

   - Replace `MarketplaceMonitor` sync methods with async equivalents
   - Convert `schedule` library usage to `asyncio` task scheduling
   - Implement async context managers for browser lifecycle
   - Add concurrent task management for multiple searches

3. **Marketplace Integration (`facebook.py`)**

   - Migrate from `playwright.sync_api` to `playwright.async_api`
   - Convert page navigation and element interactions to async
   - Implement async context managers for browser/page lifecycle
   - Add concurrent listing processing capabilities

4. **AI Backend (`ai.py`)**

   - Replace synchronous OpenAI client with async client
   - Implement async retry logic and error handling
   - Enable concurrent AI evaluations across multiple listings
   - Add async caching operations

5. **Notification System (`notification.py`)**

   - Convert all notification providers to async implementations
   - Implement async HTTP clients for notification APIs
   - Enable concurrent notification sending to multiple users
   - Add async retry mechanisms

6. **Utility Functions (`utils.py`)**
   - Replace `time.sleep()` with `asyncio.sleep()`
   - Convert file I/O operations to async where beneficial
   - Implement async versions of utility functions

## Data Models

- **Listing Model**: No changes required - dataclass remains synchronous
- **Configuration**: Maintain existing TOML-based config system
- **Cache Operations**: Evaluate migration to async cache operations for performance

## APIs and Integrations

- **Facebook Marketplace**: Async web scraping with Playwright async API
- **OpenAI API**: Async client for concurrent AI evaluations
- **Notification APIs**: Async HTTP clients for all notification providers
- **File System**: Async file operations where beneficial

## Infrastructure Requirements

- **Python 3.10+**: Required for modern async/await syntax and asyncio features
- **Async Library Dependencies**:
  - `playwright` (async API)
  - `aiohttp` or `httpx` for async HTTP requests
  - `aiofiles` for async file operations (if needed)
- **Testing Infrastructure**: Async-compatible test framework setup

# Development Roadmap

## Phase 0: Technical Spike & Foundation

**Scope**: Validate critical async patterns and establish test infrastructure

### Critical Technical Validations
- **CLI + AsyncIO Integration**: Verify Typer CLI works properly with `asyncio.run()` wrapper
- **Browser Lifecycle Proof**: Create minimal async browser context manager that preserves current sharing patterns
- **Cache Operation Analysis**: Test `diskcache` blocking behavior in async context, determine if acceptable or needs async wrapper
- **Schedule Migration**: Prototype asyncio task scheduling to replicate current interval-based patterns
- **Config Reload Integration**: Verify file watching (`watchdog`) works with async event loop

### Async Test Infrastructure Setup
- Add `pytest-asyncio` and configure for async test execution
- Create async versions of core fixtures (`async_browser`, `async_monitor`)
- Establish pattern for running sync and async tests in parallel during migration
- **Critical**: Async tests must be working before any implementation conversion

### Simple Success Validation
- Existing functionality works identically with async foundation
- No performance degradation from sync version
- Browser automation behaves identically (same pages scraped, same data extracted)
- All external integrations function unchanged (AI, notifications, config)

## Phase 1: Async Foundation with Sequential Execution

**Scope**: Minimal async wrapper while preserving exact sequential behavior

- Convert CLI entry point to use `asyncio.run()` wrapper
- Convert `MarketplaceMonitor` main loop to async (keeping exact sequential execution)
- Replace `time.sleep()` with `asyncio.sleep()`
- Update utility functions to async versions
- **Key Constraint**: No concurrent operations - pure sequential async conversion
- **Validation**: Async tests pass, behavior identical to sync version

## Phase 2: Browser Operations Migration

**Scope**: Convert Playwright to async API while preserving exact scraping behavior

- Migrate from `playwright.sync_api` to `playwright.async_api`
- Convert browser/page lifecycle to async context managers
- Update all page interactions to async patterns
- Preserve identical scraping logic, error handling, and retry patterns
- **Key Constraint**: Same pages scraped, same data extracted, same error handling
- **Validation**: Browser automation tests pass with async Playwright

## Phase 3: External API Migration

**Scope**: Convert HTTP clients to async while preserving behavior

- Migrate OpenAI client to async version
- Convert notification HTTP requests to async
- Update retry and error handling for async patterns
- Address cache operations: keep sync or add async wrapper based on Phase 0 analysis
- **Key Constraint**: Same API calls, same responses, same error handling
- **Validation**: AI evaluation and notification tests pass

## Phase 4: Integration Validation

**Scope**: End-to-end validation of complete async system

- Run full test suite in async mode
- Validate keyboard interrupt handling with async event loop
- Verify config reloading works with async main loop
- Test complete monitoring cycles (startup → search → AI → notify → repeat)
- **Final Validation**: System behaves identically to sync version with async foundation

# Logical Dependency Chain

## KISS Principle: Async Migration Strategy

**Core Philosophy**: Convert to async patterns while changing as little behavior as possible

1. **Phase 0**: Validate critical patterns work - tests must come first
2. **Phase 1**: Minimal async wrapper - just event loop, no behavior changes
3. **Phase 2**: Browser async conversion - most critical for library compatibility
4. **Phase 3**: API client conversion - complete async foundation
5. **Phase 4**: Integration validation - confirm identical behavior

**Key Constraints Throughout**:
- **No Concurrency**: Pure sequential async conversion only
- **Identical Behavior**: Same inputs produce same outputs
- **Test-First**: Async tests working before implementation changes
- **Rollback Ready**: Each phase can revert to previous working state

## Risk Mitigation Through Simplicity

- Preserve existing sequential execution order during initial migration
- Maintain identical error handling and logging behavior
- Keep complex concurrent patterns out of scope for this migration
- Focus on clean, readable async/await patterns over performance optimization

# Risks and Mitigations

## Technical Challenges

### Critical Risk: CLI + AsyncIO Integration

- **Issue**: Typer CLI framework integration with `asyncio.run()` wrapper has unknown interactions
- **Mitigation**: Phase 0 validation with simple CLI + async proof-of-concept
- **Rollback**: Revert to sync version if CLI integration issues discovered

### Critical Risk: Browser Context Management in Async

- **Issue**: Current browser sharing patterns may not work with async context managers
- **Mitigation**: Phase 0 creates minimal async browser lifecycle proof that preserves sharing
- **Rollback**: Keep sync browser management if async version introduces instability

### Critical Risk: Cache Operations Blocking Async Event Loop

- **Issue**: `diskcache` synchronous operations may block async event loop significantly
- **Mitigation**: Phase 0 tests cache operation blocking time and determines acceptable threshold
- **Alternative**: Add async wrapper for cache operations if blocking is problematic

### Risk: Schedule Library Migration Complexity

- **Issue**: Current `schedule` library has sophisticated interval management
- **Mitigation**: Phase 0 prototypes asyncio task scheduling for current patterns
- **Alternative**: Keep `schedule` library with async wrapper if direct migration too complex

### Risk: File Watching + Async Event Loop Conflicts

- **Issue**: Configuration file watching (`watchdog`) integration with asyncio unknown
- **Mitigation**: Phase 0 tests config reload with async event loop
- **Fallback**: Disable automatic config reload if conflicts discovered

## MVP Scope Definition (Simplified)

### Minimum Viable Async System

- Same CLI interface and user experience
- Async event loop foundation established
- All I/O operations converted to async (Playwright, HTTP clients)
- Preserved sequential execution behavior
- Same error handling and logging patterns

### Success Criteria (Simple & Measurable)

- **Functional Equivalence**: Same CLI behavior, same scraping results, same AI evaluations, same notifications
- **Test Coverage**: All async tests pass with identical assertions to sync tests
- **Performance**: No slower than sync version for single-threaded execution
- **Library Compatibility**: Can use async-first libraries (main goal achieved)
- **Code Quality**: Async patterns follow established Python async conventions

## Resource Constraints

### Development Time

- **Risk**: Async migration requires significant refactoring effort
- **Mitigation**: Phased approach allows for incremental progress and testing

### Learning Curve

- **Risk**: Team familiarity with async patterns may be limited
- **Mitigation**: Start with simple async conversions and build expertise gradually

### Testing Overhead

- **Risk**: Async testing requires new patterns and potentially new tools
- **Mitigation**: Invest in async testing infrastructure early in Phase 1

# Appendix

## Research Findings

- **Playwright Async Performance**: Async API shows 2-3x improvement in concurrent scenarios
- **OpenAI Async Client**: Native async support available, well-documented migration path
- **Python Async Ecosystem**: Mature libraries available for all current dependencies

## Technical Specifications

### Library Dependencies (Minimal Changes)

```toml
# Existing dependencies that already support async
playwright = ">=1.41.0"  # Already has async API available
openai = ">=1.24.0"      # Already has async client support

# Testing dependency additions
pytest-asyncio = ">=0.21.0"  # For async test execution

# No new HTTP client dependencies needed - OpenAI client handles async HTTP
# No new file I/O dependencies needed - minimal async file operations required
```

### Current Testing Framework Analysis

The project uses:

- **pytest** as the main testing framework
- **pytest-playwright** for browser automation testing
- **Sync Playwright API** throughout test suite
- **Standard pytest fixtures** in `conftest.py`
- **Parametrized tests** for configuration validation
- **No async test patterns** currently implemented

### Required Testing Changes

- Add `pytest-asyncio` plugin to support async test execution
- Convert Playwright tests from sync to async API incrementally
- Update fixtures to provide async context managers
- Maintain identical test coverage and assertions
- Preserve existing test data and HTML fixtures

### Error Handling Strategy

- Keep existing try/catch patterns around browser operations
- Add async-compatible retry mechanisms for HTTP clients
- Maintain current logging patterns (Rich library is async-compatible)
- Preserve keyboard interrupt handling with asyncio event loop

### Backward Compatibility

- CLI interface remains unchanged for end users
- Configuration files maintain same format and structure
- Existing caching and logging functionality preserved
- Graceful degradation to sequential processing if needed
