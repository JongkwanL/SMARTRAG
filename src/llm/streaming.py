"""
Streaming response handling for LLM clients.

This module provides utilities for handling streaming responses from LLM services,
including buffering, token counting, and response aggregation.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """Metrics for streaming responses."""
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    total_tokens: int = 0
    chunks_received: int = 0
    bytes_received: int = 0
    
    @property
    def total_time(self) -> Optional[float]:
        """Total generation time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.first_token_time:
            return self.first_token_time - self.start_time
        return None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Tokens per second generation rate."""
        if self.total_time and self.total_time > 0:
            return self.total_tokens / self.total_time
        return None


@dataclass
class StreamingResponse:
    """Container for streaming response data."""
    content: str = ""
    metrics: StreamingMetrics = field(default_factory=StreamingMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_chunk(self, chunk: str) -> None:
        """Add a content chunk to the response."""
        if not self.content and not self.metrics.first_token_time:
            self.metrics.first_token_time = time.time()
        
        self.content += chunk
        self.metrics.chunks_received += 1
        self.metrics.bytes_received += len(chunk.encode('utf-8'))
        
        # Rough token estimation (1 token ≈ 4 characters)
        self.metrics.total_tokens = len(self.content) // 4
    
    def finalize(self) -> None:
        """Finalize the streaming response."""
        self.metrics.end_time = time.time()


class StreamingBuffer:
    """Buffer for streaming content with configurable flushing."""
    
    def __init__(
        self,
        flush_callback: Optional[Callable[[str], None]] = None,
        buffer_size: int = 1024,
        flush_interval: float = 0.1,
        word_boundary: bool = True
    ):
        """
        Initialize streaming buffer.
        
        Args:
            flush_callback: Function to call when flushing buffer
            buffer_size: Maximum buffer size before automatic flush
            flush_interval: Time interval for automatic flush (seconds)
            word_boundary: Whether to respect word boundaries when flushing
        """
        self.flush_callback = flush_callback
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.word_boundary = word_boundary
        
        self._buffer = ""
        self._last_flush = time.time()
        self._flush_task: Optional[asyncio.Task] = None
        
        logger.debug(f"Initialized streaming buffer with size {buffer_size}")
    
    async def add(self, chunk: str) -> None:
        """Add content chunk to buffer."""
        self._buffer += chunk
        
        # Check if we should flush
        current_time = time.time()
        should_flush = (
            len(self._buffer) >= self.buffer_size or
            (current_time - self._last_flush) >= self.flush_interval
        )
        
        if should_flush:
            await self._flush()
    
    async def _flush(self) -> None:
        """Flush the buffer content."""
        if not self._buffer:
            return
        
        content_to_flush = self._buffer
        
        # Respect word boundaries if enabled
        if self.word_boundary and len(self._buffer) < self.buffer_size:
            # Keep partial words in buffer
            words = self._buffer.rsplit(' ', 1)
            if len(words) > 1:
                content_to_flush = words[0] + ' '
                self._buffer = words[1]
            else:
                # No complete words, flush everything
                self._buffer = ""
        else:
            self._buffer = ""
        
        if content_to_flush and self.flush_callback:
            self.flush_callback(content_to_flush)
        
        self._last_flush = time.time()
    
    async def finalize(self) -> None:
        """Flush any remaining content and cleanup."""
        if self._buffer:
            if self.flush_callback:
                self.flush_callback(self._buffer)
            self._buffer = ""
        
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass


class StreamingHandler:
    """Handler for processing streaming LLM responses."""
    
    def __init__(
        self,
        chunk_callback: Optional[Callable[[str], None]] = None,
        completion_callback: Optional[Callable[[StreamingResponse], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        buffer_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize streaming handler.
        
        Args:
            chunk_callback: Called for each content chunk
            completion_callback: Called when streaming completes
            error_callback: Called when an error occurs
            buffer_config: Configuration for streaming buffer
        """
        self.chunk_callback = chunk_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        
        buffer_config = buffer_config or {}
        self.buffer = StreamingBuffer(
            flush_callback=self._on_buffer_flush,
            **buffer_config
        )
        
        self.response = StreamingResponse()
        self._is_complete = False
        
        logger.debug("Initialized streaming handler")
    
    def _on_buffer_flush(self, content: str) -> None:
        """Called when buffer flushes content."""
        if self.chunk_callback:
            self.chunk_callback(content)
    
    async def process_stream(
        self,
        stream: AsyncGenerator[str, None]
    ) -> StreamingResponse:
        """
        Process a streaming generator.
        
        Args:
            stream: Async generator yielding content chunks
            
        Returns:
            StreamingResponse with complete content and metrics
        """
        try:
            async for chunk in stream:
                await self.add_chunk(chunk)
            
            await self.finalize()
            return self.response
            
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            if self.error_callback:
                self.error_callback(e)
            raise
    
    async def add_chunk(self, chunk: str) -> None:
        """Add a content chunk to the stream."""
        if self._is_complete:
            logger.warning("Received chunk after stream completion")
            return
        
        self.response.add_chunk(chunk)
        await self.buffer.add(chunk)
    
    async def finalize(self) -> None:
        """Finalize the streaming response."""
        if self._is_complete:
            return
        
        await self.buffer.finalize()
        self.response.finalize()
        self._is_complete = True
        
        if self.completion_callback:
            self.completion_callback(self.response)
        
        logger.info(
            f"Streaming completed: {self.response.metrics.total_tokens} tokens "
            f"in {self.response.metrics.total_time:.2f}s "
            f"({self.response.metrics.tokens_per_second:.1f} tokens/s)"
        )


class StreamingAggregator:
    """Aggregates multiple streaming responses."""
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize streaming aggregator.
        
        Args:
            max_concurrent: Maximum concurrent streams to process
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.debug(f"Initialized streaming aggregator with {max_concurrent} max concurrent")
    
    async def aggregate_streams(
        self,
        streams: List[AsyncGenerator[str, None]],
        merge_strategy: str = "concat"
    ) -> StreamingResponse:
        """
        Aggregate multiple streaming responses.
        
        Args:
            streams: List of streaming generators
            merge_strategy: Strategy for merging responses ("concat", "interleave")
            
        Returns:
            Aggregated StreamingResponse
        """
        if not streams:
            return StreamingResponse()
        
        if merge_strategy == "concat":
            return await self._concat_streams(streams)
        elif merge_strategy == "interleave":
            return await self._interleave_streams(streams)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    
    async def _concat_streams(
        self,
        streams: List[AsyncGenerator[str, None]]
    ) -> StreamingResponse:
        """Concatenate streams sequentially."""
        aggregated_response = StreamingResponse()
        
        for stream in streams:
            async with self.semaphore:
                async for chunk in stream:
                    aggregated_response.add_chunk(chunk)
        
        aggregated_response.finalize()
        return aggregated_response
    
    async def _interleave_streams(
        self,
        streams: List[AsyncGenerator[str, None]]
    ) -> StreamingResponse:
        """Interleave streams by alternating chunks."""
        aggregated_response = StreamingResponse()
        active_streams = list(enumerate(streams))
        chunk_queue = deque()
        
        async def collect_chunks():
            """Collect chunks from all streams."""
            tasks = []
            for i, stream in active_streams:
                task = asyncio.create_task(self._get_next_chunk(stream, i))
                tasks.append(task)
            
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        chunk, stream_idx = await task
                        if chunk is not None:
                            chunk_queue.append((stream_idx, chunk))
                            # Create new task for this stream
                            new_task = asyncio.create_task(
                                self._get_next_chunk(
                                    active_streams[stream_idx][1], 
                                    stream_idx
                                )
                            )
                            tasks.append(new_task)
                    except StopAsyncIteration:
                        # Stream is exhausted
                        pass
                
                tasks = list(pending)
        
        # Start collecting chunks
        collect_task = asyncio.create_task(collect_chunks())
        
        # Process chunks as they arrive
        while not collect_task.done() or chunk_queue:
            if chunk_queue:
                stream_idx, chunk = chunk_queue.popleft()
                aggregated_response.add_chunk(chunk)
            else:
                await asyncio.sleep(0.01)  # Brief pause
        
        await collect_task
        aggregated_response.finalize()
        return aggregated_response
    
    async def _get_next_chunk(
        self,
        stream: AsyncGenerator[str, None],
        stream_idx: int
    ) -> tuple[Optional[str], int]:
        """Get next chunk from stream."""
        try:
            chunk = await stream.__anext__()
            return chunk, stream_idx
        except StopAsyncIteration:
            return None, stream_idx


# Utility functions for common streaming patterns

async def collect_stream(stream: AsyncGenerator[str, None]) -> str:
    """Collect all chunks from a stream into a single string."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


async def tee_stream(
    stream: AsyncGenerator[str, None],
    num_outputs: int = 2
) -> List[AsyncGenerator[str, None]]:
    """Split a stream into multiple output streams."""
    if num_outputs < 1:
        raise ValueError("num_outputs must be >= 1")
    
    queues = [asyncio.Queue() for _ in range(num_outputs)]
    
    async def producer():
        """Produce chunks to all queues."""
        try:
            async for chunk in stream:
                for queue in queues:
                    await queue.put(chunk)
        finally:
            # Signal end of stream
            for queue in queues:
                await queue.put(None)
    
    async def consumer(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Consume chunks from a queue."""
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    
    # Start producer task
    producer_task = asyncio.create_task(producer())
    
    # Return consumer generators
    consumers = [consumer(queue) for queue in queues]
    
    # Cleanup when all consumers are done
    async def cleanup():
        await producer_task
    
    # Attach cleanup to first consumer (simplification)
    if consumers:
        original_aclose = consumers[0].aclose
        
        async def aclose_with_cleanup():
            await cleanup()
            if hasattr(original_aclose, '__call__'):
                await original_aclose()
        
        consumers[0].aclose = aclose_with_cleanup
    
    return consumers