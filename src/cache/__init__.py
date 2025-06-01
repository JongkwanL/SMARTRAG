"""
Cache module for SmartRAG.

This module provides Redis-based caching capabilities with various strategies
including TTL, LRU, and custom eviction policies.
"""

from .redis_client import RedisClient, RedisPool
from .strategies import (
    CacheStrategy,
    TTLStrategy,
    LRUStrategy,
    LFUStrategy,
    FIFOStrategy,
    CacheManager
)

__all__ = [
    "RedisClient",
    "RedisPool", 
    "CacheStrategy",
    "TTLStrategy",
    "LRUStrategy",
    "LFUStrategy",
    "FIFOStrategy",
    "CacheManager",
]