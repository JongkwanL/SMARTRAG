"""
Redis client implementation for SmartRAG caching.

This module provides async Redis client with connection pooling, 
retry logic, and health monitoring.
"""

import asyncio
import logging
import pickle
import json
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from ..core.config import get_settings
from ..core.exceptions import CacheError, CacheConnectionError, CacheTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = None
    health_check_interval: int = 30
    retry_on_timeout: bool = True
    retry_on_error: List[Exception] = None
    max_retries: int = 3
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {}
        if self.retry_on_error is None:
            self.retry_on_error = [
                redis.ConnectionError,
                redis.TimeoutError,
                redis.BusyLoadingError,
            ]


class RedisPool:
    """Redis connection pool manager."""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis pool.
        
        Args:
            config: Redis configuration
        """
        self.config = config or RedisConfig()
        self.pool: Optional[ConnectionPool] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = True
        
        logger.debug(f"Initialized Redis pool config: {self.config.host}:{self.config.port}")
    
    async def create_pool(self) -> ConnectionPool:
        """Create Redis connection pool."""
        if self.pool is not None:
            return self.pool
        
        retry = Retry(
            ExponentialBackoff(),
            self.config.max_retries
        ) if self.config.retry_on_timeout else None
        
        self.pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            socket_keepalive=self.config.socket_keepalive,
            socket_keepalive_options=self.config.socket_keepalive_options,
            retry=retry,
            decode_responses=False,  # We handle encoding/decoding manually
        )
        
        # Start health check
        if self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Created Redis pool: {self.config.host}:{self.config.port}")
        return self.pool
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client from pool."""
        if self.pool is None:
            await self.create_pool()
        
        return redis.Redis(connection_pool=self.pool)
    
    async def close(self):
        """Close the connection pool."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.pool:
            await self.pool.disconnect()
            self.pool = None
        
        logger.info("Closed Redis pool")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                client = await self.get_client()
                start_time = time.time()
                await client.ping()
                latency = time.time() - start_time
                
                self._is_healthy = True
                logger.debug(f"Redis health check OK (latency: {latency:.3f}s)")
                
                await client.close()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._is_healthy = False
                logger.warning(f"Redis health check failed: {e}")
    
    @property
    def is_healthy(self) -> bool:
        """Check if Redis is healthy."""
        return self._is_healthy


class RedisClient:
    """Async Redis client with serialization and error handling."""
    
    def __init__(
        self,
        pool: Optional[RedisPool] = None,
        serializer: str = "pickle",
        key_prefix: str = "smartrag:",
        compression: bool = False
    ):
        """
        Initialize Redis client.
        
        Args:
            pool: Redis pool instance
            serializer: Serialization method ("pickle", "json")
            key_prefix: Prefix for all keys
            compression: Whether to use compression
        """
        settings = get_settings()
        
        if pool is None:
            config = RedisConfig(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                max_connections=settings.redis_max_connections,
            )
            pool = RedisPool(config)
        
        self.pool = pool
        self.serializer = serializer
        self.key_prefix = key_prefix
        self.compression = compression
        
        # Performance metrics
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
        
        logger.info(f"Initialized Redis client with serializer: {serializer}")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.serializer == "pickle":
                data = pickle.dumps(value)
            elif self.serializer == "json":
                data = json.dumps(value, default=str).encode('utf-8')
            else:
                raise ValueError(f"Unknown serializer: {self.serializer}")
            
            if self.compression:
                import gzip
                data = gzip.compress(data)
            
            return data
            
        except Exception as e:
            raise CacheError(f"Serialization failed: {e}")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if self.compression:
                import gzip
                data = gzip.decompress(data)
            
            if self.serializer == "pickle":
                return pickle.loads(data)
            elif self.serializer == "json":
                return json.loads(data.decode('utf-8'))
            else:
                raise ValueError(f"Unknown serializer: {self.serializer}")
                
        except Exception as e:
            raise CacheError(f"Deserialization failed: {e}")
    
    async def _execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute Redis operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.pool.config.max_retries + 1):
            try:
                client = await self.pool.get_client()
                result = await operation(client, *args, **kwargs)
                await client.close()
                return result
                
            except redis.ConnectionError as e:
                last_exception = CacheConnectionError(f"Redis connection error: {e}")
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                
            except redis.TimeoutError as e:
                last_exception = CacheTimeoutError(f"Redis timeout: {e}")
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_exception = CacheError(f"Redis operation failed: {e}")
                logger.error(f"Redis error on attempt {attempt + 1}: {e}")
                self.error_count += 1
                
                # Don't retry on serialization errors
                if "Serialization" in str(e) or "Deserialization" in str(e):
                    break
            
            # Wait before retry
            if attempt < self.pool.config.max_retries:
                delay = 2 ** attempt * 0.1  # Exponential backoff
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        async def _get(client: redis.Redis) -> Optional[Any]:
            prefixed_key = self._make_key(key)
            data = await client.get(prefixed_key)
            
            if data is None:
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            return self._deserialize(data)
        
        try:
            return await self._execute_with_retry(_get)
        except Exception as e:
            logger.error(f"Failed to get key '{key}': {e}")
            self.miss_count += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if value was set
        """
        async def _set(client: redis.Redis) -> bool:
            prefixed_key = self._make_key(key)
            data = self._serialize(value)
            
            result = await client.set(
                prefixed_key,
                data,
                ex=ttl,
                nx=nx,
                xx=xx
            )
            
            return bool(result)
        
        return await self._execute_with_retry(_set)
    
    async def delete(self, *keys: str) -> int:
        """
        Delete keys from cache.
        
        Args:
            keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        async def _delete(client: redis.Redis) -> int:
            prefixed_keys = [self._make_key(key) for key in keys]
            return await client.delete(*prefixed_keys)
        
        return await self._execute_with_retry(_delete)
    
    async def exists(self, *keys: str) -> int:
        """
        Check if keys exist.
        
        Args:
            keys: Keys to check
            
        Returns:
            Number of existing keys
        """
        async def _exists(client: redis.Redis) -> int:
            prefixed_keys = [self._make_key(key) for key in keys]
            return await client.exists(*prefixed_keys)
        
        return await self._execute_with_retry(_exists)
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if TTL was set
        """
        async def _expire(client: redis.Redis) -> bool:
            prefixed_key = self._make_key(key)
            return bool(await client.expire(prefixed_key, ttl))
        
        return await self._execute_with_retry(_expire)
    
    async def ttl(self, key: str) -> int:
        """
        Get TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        async def _ttl(client: redis.Redis) -> int:
            prefixed_key = self._make_key(key)
            return await client.ttl(prefixed_key)
        
        return await self._execute_with_retry(_ttl)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment counter.
        
        Args:
            key: Counter key
            amount: Increment amount
            
        Returns:
            New counter value
        """
        async def _incr(client: redis.Redis) -> int:
            prefixed_key = self._make_key(key)
            return await client.incrby(prefixed_key, amount)
        
        return await self._execute_with_retry(_incr)
    
    async def get_multi(self, *keys: str) -> List[Optional[Any]]:
        """
        Get multiple values.
        
        Args:
            keys: Keys to get
            
        Returns:
            List of values (None for missing keys)
        """
        async def _mget(client: redis.Redis) -> List[Optional[Any]]:
            prefixed_keys = [self._make_key(key) for key in keys]
            data_list = await client.mget(prefixed_keys)
            
            results = []
            for data in data_list:
                if data is None:
                    self.miss_count += 1
                    results.append(None)
                else:
                    self.hit_count += 1
                    results.append(self._deserialize(data))
            
            return results
        
        try:
            return await self._execute_with_retry(_mget)
        except Exception as e:
            logger.error(f"Failed to get multiple keys: {e}")
            self.miss_count += len(keys)
            return [None] * len(keys)
    
    async def set_multi(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple values.
        
        Args:
            mapping: Key-value mapping
            ttl: Time to live in seconds
            
        Returns:
            True if all values were set
        """
        async def _mset(client: redis.Redis) -> bool:
            serialized_mapping = {
                self._make_key(key): self._serialize(value)
                for key, value in mapping.items()
            }
            
            if ttl is None:
                return bool(await client.mset(serialized_mapping))
            else:
                # Use pipeline for atomic TTL setting
                async with client.pipeline() as pipe:
                    await pipe.mset(serialized_mapping)
                    for key in serialized_mapping.keys():
                        await pipe.expire(key, ttl)
                    results = await pipe.execute()
                    return all(results)
        
        return await self._execute_with_retry(_mset)
    
    async def flush_db(self) -> bool:
        """
        Flush database (delete all keys).
        
        Returns:
            True if successful
        """
        async def _flush(client: redis.Redis) -> bool:
            await client.flushdb()
            return True
        
        logger.warning("Flushing Redis database")
        return await self._execute_with_retry(_flush)
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys (without prefix)
        """
        async def _keys(client: redis.Redis) -> List[str]:
            prefixed_pattern = self._make_key(pattern)
            keys = await client.keys(prefixed_pattern)
            
            # Remove prefix from keys
            prefix_len = len(self.key_prefix)
            return [key.decode('utf-8')[prefix_len:] for key in keys]
        
        return await self._execute_with_retry(_keys)
    
    async def info(self) -> Dict[str, Any]:
        """
        Get Redis server info.
        
        Returns:
            Server info dictionary
        """
        async def _info(client: redis.Redis) -> Dict[str, Any]:
            return await client.info()
        
        return await self._execute_with_retry(_info)
    
    async def ping(self) -> bool:
        """
        Ping Redis server.
        
        Returns:
            True if server is responding
        """
        async def _ping(client: redis.Redis) -> bool:
            return bool(await client.ping())
        
        try:
            return await self._execute_with_retry(_ping)
        except Exception:
            return False
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "error_count": self.error_count,
            "hit_rate": self.hit_rate,
            "is_healthy": self.pool.is_healthy,
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
    
    async def close(self):
        """Close Redis client."""
        await self.pool.close()