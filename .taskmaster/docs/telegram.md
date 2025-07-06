# Telegram Notifications for AI Marketplace Monitor - PRD

<context>
# Overview
This PRD outlines the implementation of Telegram notifications for the AI Marketplace Monitor project. The goal is to integrate Telegram as a notification channel alongside existing options (email, Pushover, Pushbullet, Ntfy) to provide users with unlimited message length and rich formatting capabilities for marketplace listing alerts.

# Core Features
- **Telegram Bot Integration**: Leverage python-telegram-bot library for reliable message delivery
- **Configuration Management**: Simple bot token and chat ID setup in user config
- **Message Formatting**: Support for Telegram's native MarkdownV2 and HTML formatting
- **Long Message Handling**: Automatic message splitting for content exceeding Telegram's 4096 character limit
- **Error Handling**: Standard retry logic with exponential backoff for network failures
- **Testing**: Comprehensive unit test coverage with mocked Telegram API calls

# User Experience
- **Setup**: Users provide bot token and chat ID in configuration file
- **Notifications**: Receive rich-formatted marketplace alerts via Telegram
- **Reliability**: Automatic retry on transient failures, fallback behavior consistent with other notification types
- **Content**: Full listing details including AI ratings, descriptions, and direct links
</context>

<PRD>
# Technical Architecture

## System Components

### TelegramNotificationConfig Class
- Inherits from `NotificationConfig` base class
- Fields:
  - `telegram_bot_token: str` - Bot authentication token
  - `telegram_chat_id: str` - Target chat/channel ID
  - `message_format: str` - Format type ('markdownv2', 'html', 'text')
- Implements `send_message()` method for notification delivery

### Message Processing Pipeline
1. **Format Selection**: Use configured message format (default: 'markdownv2')
2. **Content Preparation**: Leverage existing notification content from base classes
3. **Message Splitting**: Implement chunking for messages > 4096 characters
4. **Delivery**: Send via python-telegram-bot with retry logic
5. **Status Tracking**: Use existing NotificationStatus enum

### Integration Points
- Add TelegramNotificationConfig to UserConfig inheritance chain
- Utilize existing notification orchestration in user.py
- Leverage established configuration validation patterns
- Follow existing dataclass-based configuration structure

## Data Models

### Configuration Schema
```toml
[notification.telegram]
telegram_bot_token = "bot_token_here"
telegram_chat_id = "chat_id_here"
message_format = "markdownv2"  # options: markdownv2, html, text
```

### Message Structure
- Reuse existing notification content structure
- Support all current listing fields (title, price, location, AI rating, etc.)
- Maintain status indicators (NEW, UPDATED, DISCOUNTED, etc.)

## APIs and Integrations

### python-telegram-bot Library
- Use latest python-telegram-bot but treat it as synchronous
- Wrap ALL async methods with `asyncio.run()` to maintain synchronous interface
- Primary methods: `asyncio.run(bot.send_message())` with parse_mode parameter
- NO async/await keywords anywhere in our code - maintain pure synchronous simplicity
- Error handling: Catch `telegram.error.NetworkError` for retry logic
- Project-wide synchronous approach for testing simplicity and code consistency

### Message Formatting Support
- **MarkdownV2**: Native Telegram markdown (default)
- **HTML**: Rich text formatting with links and bold/italic
- **Text**: Plain text fallback

## Infrastructure Requirements

### Dependencies
- Add `python-telegram-bot` using `poetry add python-telegram-bot`
- Use existing pydantic2 for any custom types or data validation
- No additional infrastructure requirements (leverages existing config system)

### Testing Infrastructure
- Mock `telegram.Bot` class for unit tests
- Test message splitting logic with various content lengths
- Validate retry behavior with simulated network failures

# Development Roadmap

## MVP Phase
1. **Core Implementation**
   - Create TelegramNotificationConfig class
   - Implement basic send_message() functionality
   - Add configuration validation
   - Integrate with existing UserConfig system

2. **Message Handling**
   - Implement message splitting for long content
   - Support all three format types (markdownv2, html, text)
   - Error handling with standard retry logic

3. **Testing Foundation**
   - Unit tests for TelegramNotificationConfig
   - Mock-based testing for all Telegram API interactions
   - Test coverage for message splitting edge cases
   - Retry logic validation tests

## Future Enhancements (Post-MVP)
- Interactive buttons (callback handlers)
- Media attachment support (images)
- Multiple chat/channel support per user
- Advanced formatting templates
- Rate limiting optimization

# Logical Dependency Chain

## Foundation Requirements (Build First)
1. **TelegramNotificationConfig Class**: Core notification implementation
2. **Configuration Integration**: Add to UserConfig inheritance chain
3. **Basic Message Sending**: Simple text message delivery

## Incremental Additions (Build Upon)
1. **Message Formatting**: Add MarkdownV2/HTML support
2. **Message Splitting**: Handle long content gracefully
3. **Error Handling**: Implement retry logic with exponential backoff
4. **Comprehensive Testing**: Full unit test coverage

## Integration Completion
1. **Documentation Updates**: Configuration examples and setup guide
2. **Validation**: End-to-end testing with existing notification system
3. **Deployment**: Integration with main notification pipeline

# Risks and Mitigations

## Technical Challenges
- **Maintaining Synchronous Simplicity**: Consistently wrap ALL async calls with asyncio.run() to avoid any async/await in our codebase
- **Message Formatting Complexity**: MarkdownV2 has strict escaping rules; provide fallback to HTML/text
- **Rate Limiting**: Telegram has rate limits; implement exponential backoff retry logic

## MVP Scope Management
- **Keep Simple**: Focus on basic message delivery without interactive features
- **Leverage Library**: Use python-telegram-bot built-in capabilities rather than reimplementing
- **Consistent Patterns**: Follow existing notification class patterns for easier maintenance

## Resource Constraints
- **Testing Complexity**: Mock all Telegram API calls to avoid external dependencies
- **Configuration Simplicity**: Reuse existing config patterns rather than creating new structures
- **Error Scenarios**: Handle network failures gracefully with standard retry patterns

# Appendix

## Research Findings
- python-telegram-bot v20+ is async-only but we treat it as synchronous with asyncio.run() wrapper
- Complete project synchronous approach for simplicity and testing ease
- Telegram message limit is 4096 characters, requiring manual message splitting
- Library provides robust error handling for NetworkError exceptions
- MarkdownV2, HTML, and text formatting are natively supported
- No built-in retry logic - must implement custom retry wrapper

## Technical Specifications
- **Message Splitting**: Chunk at word boundaries when possible, max 4096 chars per chunk
- **Retry Logic**: Exponential backoff (0.1s, 0.2s, 0.4s, 0.8s, 1.6s) for max 5 attempts
- **Format Priority**: markdownv2 > html > text (graceful degradation on format errors)
- **Configuration Validation**: Validate bot token format and chat ID format during config load

## Implementation Notes
- Follow red/green TDD approach for all new code
- Use latest python-telegram-bot but maintain 100% synchronous codebase with asyncio.run() wrapper
- Use pydantic2 for any custom types, data models, or validation (consistent with project patterns)
- Mock telegram.Bot class completely for unit tests (no async test complexity)
- Use existing notification base class patterns for consistency
- Integrate with existing error logging and monitoring systems
- Critical: NO async/await keywords in our implementation - treat library as synchronous
</PRD>
