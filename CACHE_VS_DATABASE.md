# Cache vs Database Storage Architecture

## Overview

The Knowledge App uses **two separate storage systems** for questions, each serving a different purpose:

## 🗄️ Database Storage (Permanent)

**Purpose**: Permanent storage for review history and user progress tracking

**Location**: `user_data/question_history.sqlite`

**What it stores**:
- All generated questions with full metadata
- Quiz session information
- User answer history and performance
- Question difficulty, topic, and generation method
- Timestamps for when questions were generated and answered

**Retention**: Permanent (until user manually clears)

**Used for**:
- ✅ Review Question History feature
- ✅ User progress tracking
- ✅ Preventing duplicate questions across sessions
- ✅ Analytics and learning insights

## ⚡ Cache Storage (Temporary)

**Purpose**: Short-term performance optimization to avoid redundant API calls

**Location**: `data/cache/question_cache.json`

**What it stores**:
- Unused questions from recent quiz sessions
- Only for non-expert difficulty levels
- Temporary buffer to improve response time

**Retention**: 10 minutes TTL (Time To Live)

**Used for**:
- ✅ Avoiding API calls for similar questions within 10 minutes
- ✅ Improving response time for repeated quiz parameters
- ✅ Reducing GPU/AI model usage for identical requests

## 🚫 What Cache is NOT Used For

- ❌ Long-term question storage
- ❌ Review history functionality
- ❌ Expert mode questions (always fresh)
- ❌ User progress tracking
- ❌ Cross-session question deduplication

## Expert Mode Special Handling

**Expert mode questions are NEVER cached** to ensure maximum diversity:

1. **No caching**: Expert questions bypass all cache mechanisms
2. **Fresh generation**: Each expert question is generated from scratch
3. **Database only**: Expert questions are saved to database for review history
4. **Maximum diversity**: Prevents repetition of the same question

## Flow Diagram

```
User Requests Question
         ↓
    Is Expert Mode?
         ↓
    YES → Generate Fresh → Save to Database → Show to User
         ↓
    NO → Check Cache (10 min TTL)
         ↓
    Cache Hit? → Return Cached → Show to User
         ↓
    Cache Miss → Generate Fresh → Save to Cache & Database → Show to User
```

## File Locations

```
knowledge_app/
├── user_data/
│   └── question_history.sqlite     # 🗄️ PERMANENT DATABASE
└── data/
    └── cache/
        └── question_cache.json     # ⚡ TEMPORARY CACHE (10 min TTL)
```

## Code Components

### Database Storage
- `QuestionHistoryStorage` class in `core/question_history_storage.py`
- SQLite database with comprehensive schema
- Handles question and quiz session persistence

### Cache Storage
- `QuestionCache` class in `core/question_generator.py`
- JSON file-based temporary storage
- Automatic expiration and cleanup

### Cache Management
- `_save_question_cache()` in `webengine_app.py`
- `_load_question_cache()` in `webengine_app.py`
- `_cache_matches_current_quiz()` for cache validation

## Configuration

### Cache TTL Settings
```python
# Inference layer cache
self._response_cache = ResponseCache(ttl_seconds=300)   # 5 minutes
self._quiz_cache = ResponseCache(ttl_seconds=600)       # 10 minutes

# Question generator cache
self._question_cache = QuestionCache(ttl_seconds=600)   # 10 minutes
self._response_cache = QuestionCache(ttl_seconds=300)   # 5 minutes

# Persistent cache expiration check
cache_age_minutes > 10  # 10 minute expiration
```

## Troubleshooting

### If Questions Are Repeating
1. Check if expert mode is being used (should never repeat)
2. Verify cache TTL is working (should expire after 10 minutes)
3. Check database deduplication logic
4. Clear cache manually: delete `data/cache/question_cache.json`

### If Review History Is Missing
1. Check database file exists: `user_data/question_history.sqlite`
2. Verify `_save_to_question_history()` is being called
3. Check database schema with test script
4. Look for database write errors in logs

### Cache vs Database Confusion
- **Cache**: Temporary performance optimization (10 minutes)
- **Database**: Permanent storage for review and history
- **They serve different purposes and should not be confused**

## Testing

Run the test scripts to verify both systems:

```bash
# Test database storage
python test_database_storage.py

# Test question diversity (includes cache behavior)
python test_expert_question_diversity.py
```

## Summary

The dual storage system ensures:
- ⚡ **Fast response times** through intelligent caching
- 🗄️ **Permanent question history** for review and progress tracking
- 🚫 **No expert mode caching** for maximum question diversity
- 🔄 **Clear separation** between temporary and permanent storage

This architecture solves the original issue where expert mode questions were repeating due to overly aggressive caching, while maintaining performance benefits for regular quiz modes.
