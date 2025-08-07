# Configuration Migration Guide

This document outlines the migration and deprecation strategy implemented for transitioning to the new LangChain-unified backend system while maintaining full backward compatibility.

## Overview

The AI Marketplace Monitor now uses a unified LangChain backend architecture while preserving all existing TOML configuration patterns. Users can continue using their current configurations without any changes, but the system provides helpful guidance for adopting security best practices.

## Migration Features

### 1. Automatic Warning System

The system automatically detects and warns about configuration patterns that could be improved:

- **API Keys in Config Files**: Suggests moving to environment variables for better security
- **Legacy Configuration Fields**: Warns about deprecated fields and suggests modern alternatives
- **Missing Required Fields**: Alerts about configurations that may cause runtime issues
- **Performance Settings**: Recommends optimal timeout and retry settings

### 2. Configuration Suggestions

Enable configuration suggestions by setting an environment variable:

```bash
export AI_MARKETPLACE_MONITOR_SHOW_CONFIG_TIPS=true
```

When enabled, the system provides actionable recommendations on first connection:

```
--- Configuration Suggestions for my-openai-config ---

Move API key to environment variable OPENAI_API_KEY:
  export OPENAI_API_KEY='your_api_key_here'
  # Then remove 'api_key' from your config file

Consider specifying a model for better performance:
  model = 'gpt-3.5-turbo'  # Fast and cost-effective
  # or model = 'gpt-4'     # More capable but slower/costlier

--- End Suggestions ---
```

### 3. Environment Variable Migration

**Recommended approach for API keys:**

| Provider | Environment Variable | Config File (Legacy) |
|----------|---------------------|---------------------|
| OpenAI | `OPENAI_API_KEY` | `api_key = "..."` |
| DeepSeek | `DEEPSEEK_API_KEY` | `api_key = "..."` |
| OpenRouter | `OPENROUTER_API_KEY` | `api_key = "..."` |
| Ollama | N/A (local) | N/A |

**Migration Priority:**
1. **High**: Move API keys to environment variables
2. **Medium**: Specify recommended models for better performance
3. **Low**: Optimize timeout and retry settings

### 4. Backward Compatibility Guarantees

- ✅ All existing TOML configurations work unchanged
- ✅ All provider names and parameters remain valid
- ✅ Case-insensitive provider matching preserved
- ✅ Error messages consistent with previous versions
- ✅ No breaking changes to existing workflows

## Common Migration Scenarios

### Scenario 1: Basic OpenAI Configuration
**Before:**
```toml
[ai.my_openai]
name = "my-openai"
provider = "openai"
api_key = "sk-..."
```

**After (Recommended):**
```bash
export OPENAI_API_KEY="sk-..."
```
```toml
[ai.my_openai]
name = "my-openai"
provider = "openai"
model = "gpt-3.5-turbo"  # Added for better performance
```

### Scenario 2: DeepSeek with Enhanced Settings
**Before:**
```toml
[ai.my_deepseek]
name = "my-deepseek"
provider = "deepseek"
api_key = "sk-..."
```

**After (Recommended):**
```bash
export DEEPSEEK_API_KEY="sk-..."
```
```toml
[ai.my_deepseek]
name = "my-deepseek"
provider = "deepseek"
model = "deepseek-coder"
timeout = 60
max_retries = 3
```

### Scenario 3: Ollama Local Setup
**Before:**
```toml
[ai.my_ollama]
name = "my-ollama"
provider = "ollama"
```

**After (Recommended):**
```toml
[ai.my_ollama]
name = "my-ollama"
provider = "ollama"
model = "llama2"  # Required for Ollama
base_url = "http://localhost:11434"  # Explicit for clarity
```

## Implementation Details

### Warning Categories

1. **Security Warnings**: API keys in config files
2. **Deprecation Warnings**: Legacy configuration fields
3. **Configuration Warnings**: Missing required fields or suboptimal settings
4. **Performance Warnings**: Settings that may impact performance

### Precedence Rules

When both config file and environment variables are present:

1. **Environment variables take precedence** for API keys (more secure)
2. **Config file values used** for other settings
3. **Warning logged** about mixed configuration sources

### Testing Coverage

The migration strategy includes comprehensive tests for:

- ✅ All warning scenarios and edge cases
- ✅ Configuration suggestion generation
- ✅ Environment variable precedence
- ✅ Backward compatibility validation
- ✅ Performance and thread safety

## Migration Timeline

- **Phase 1 (Current)**: Warning system active, full backward compatibility
- **Phase 2 (Future)**: Enhanced suggestions and migration tools
- **Phase 3 (Future)**: Possible deprecation of insecure patterns (with ample notice)

*Note: No breaking changes are planned. All existing configurations will continue to work indefinitely.*
