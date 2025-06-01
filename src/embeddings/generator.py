"""Embedding generation module using sentence-transformers."""

import asyncio
import hashlib
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import settings
from ..core.exceptions import EmbeddingGenerationError


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ):
        """Initialize embedding generator."""
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_seq_length = max_seq_length or settings.max_seq_length

        # Initialize model
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = self.max_seq_length

            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"

            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to load embedding model: {e}")

        # Cache for embeddings
        self._cache: Dict[str, np.ndarray] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])

        try:
            # Check cache
            embeddings = []
            texts_to_embed = []
            text_indices = []

            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append((i, self._cache[cache_key]))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)

            # Generate embeddings for uncached texts
            if texts_to_embed:
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=self.batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )

                # Add to cache and results
                for idx, text, embedding in zip(text_indices, texts_to_embed, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = embedding
                    embeddings.append((idx, embedding))

            # Sort by original index and extract embeddings
            embeddings.sort(key=lambda x: x[0])
            result = np.array([emb for _, emb in embeddings])

            return result

        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {e}")

    async def generate_embeddings_async(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_embeddings,
            texts,
            normalize,
            show_progress,
        )

    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    async def generate_embedding_async(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text asynchronously."""
        embeddings = await self.generate_embeddings_async([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """Generate embeddings in batches."""
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.generate_embeddings(batch, normalize=normalize)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def batch_generate_embeddings_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """Generate embeddings in batches asynchronously."""
        batch_size = batch_size or self.batch_size
        tasks = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            task = self.generate_embeddings_async(batch, normalize=normalize)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        all_embeddings = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """Compute similarity between two embeddings."""
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        elif metric == "euclidean":
            # Euclidean distance (negative for similarity)
            return -float(np.linalg.norm(embedding1 - embedding2))
        elif metric == "dot":
            # Dot product
            return float(np.dot(embedding1, embedding2))
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def compute_similarities(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """Compute similarities between query and multiple embeddings."""
        if len(embeddings) == 0:
            return np.array([])

        if metric == "cosine":
            # Vectorized cosine similarity
            dot_products = np.dot(embeddings, query_embedding)
            query_norm = np.linalg.norm(query_embedding)
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            
            # Avoid division by zero
            valid_indices = (embedding_norms != 0) & (query_norm != 0)
            similarities = np.zeros(len(embeddings))
            similarities[valid_indices] = dot_products[valid_indices] / (
                embedding_norms[valid_indices] * query_norm
            )
            return similarities
        elif metric == "euclidean":
            # Vectorized Euclidean distance
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            return -distances  # Negative for similarity
        elif metric == "dot":
            # Vectorized dot product
            return np.dot(embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{self.model_name}:{text_hash}"

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EmbeddingGenerator(model={self.model_name}, "
            f"device={self.device}, dim={self.embedding_dim})"
        )


@lru_cache(maxsize=1)
def get_embedding_generator() -> EmbeddingGenerator:
    """Get singleton embedding generator instance."""
    return EmbeddingGenerator()