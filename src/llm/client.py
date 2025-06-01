"""
LLM client implementations for SmartRAG.

This module provides HTTP-based clients for interacting with various LLM services,
with a focus on vLLM and OpenAI-compatible APIs.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import httpx
import time
from dataclasses import dataclass

from ..core.config import get_settings
from ..core.exceptions import LLMError, LLMTimeoutError, LLMConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        pass


class vLLMClient(LLMClient):
    """HTTP client for vLLM OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: Base URL for the vLLM API
            api_key: API key for authentication (if required)
            model: Default model name
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        settings = get_settings()
        
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key or settings.vllm_api_key
        self.model = model or settings.vllm_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure HTTP client
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30
            )
        )
        
        logger.info(f"Initialized vLLM client with base_url: {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        if hasattr(self, 'client'):
            await self.client.aclose()
    
    def _format_messages(self, messages: Union[str, List[ChatMessage]]) -> List[Dict[str, str]]:
        """Format messages for API request."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Make HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    return await self._stream_request(endpoint, payload)
                else:
                    response = await self.client.post(endpoint, json=payload)
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.TimeoutException as e:
                last_exception = LLMTimeoutError(f"Request timeout: {e}")
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                
            except httpx.ConnectError as e:
                last_exception = LLMConnectionError(f"Connection error: {e}")
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                last_exception = LLMError(error_msg)
                logger.error(f"HTTP error on attempt {attempt + 1}: {error_msg}")
                
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    break
                    
            except Exception as e:
                last_exception = LLMError(f"Unexpected error: {e}")
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (with exponential backoff)
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def _stream_request(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make streaming HTTP request."""
        payload["stream"] = True
        
        async with self.client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        import json
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming chunk: {data}")
                        continue
    
    async def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Input messages (string or list of ChatMessage)
            model: Model name to use (overrides default)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        payload = {
            "model": model or self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": False,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        
        try:
            response_data = await self._make_request("/v1/chat/completions", payload)
            
            choice = response_data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            
            usage = response_data.get("usage", {})
            
            generation_time = time.time() - start_time
            
            logger.info(
                f"Generated response in {generation_time:.2f}s "
                f"(tokens: {usage.get('total_tokens', 'unknown')})"
            )
            
            return LLMResponse(
                content=content,
                model=response_data.get("model", payload["model"]),
                usage=usage,
                finish_reason=finish_reason,
                metadata={
                    "generation_time": generation_time,
                    "request_id": response_data.get("id"),
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def stream_generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: Input messages (string or list of ChatMessage)
            model: Model name to use (overrides default)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Yields:
            Content chunks as they are generated
        """
        start_time = time.time()
        
        payload = {
            "model": model or self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": True,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        
        try:
            chunk_count = 0
            async for chunk in await self._make_request("/v1/chat/completions", payload, stream=True):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        chunk_count += 1
                        yield content
            
            generation_time = time.time() - start_time
            logger.info(
                f"Streamed {chunk_count} chunks in {generation_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False