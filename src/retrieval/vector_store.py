"""Vector store implementation for FAISS and pgvector."""

import asyncio
import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import settings
from ..core.document_processor import Chunk
from ..core.exceptions import VectorStoreError


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add embeddings and chunks to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete vectors by chunk IDs."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        index_path: Optional[str] = None,
    ):
        """Initialize FAISS vector store."""
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index_path = index_path or settings.faiss_index_path

        # Create index based on type and metric
        if metric == "cosine":
            # Normalize vectors for cosine similarity
            self.normalize = True
            if index_type == "Flat":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            elif index_type == "IVF":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
            elif index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
        elif metric == "euclidean":
            self.normalize = False
            if index_type == "Flat":
                self.index = faiss.IndexFlatL2(dimension)
            elif index_type == "IVF":
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)
            elif index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Metadata storage
        self.chunks: Dict[int, Chunk] = {}
        self.chunk_id_to_index: Dict[str, int] = {}
        self.next_id = 0

        # Train IVF index if needed
        self.is_trained = index_type != "IVF"

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add embeddings and chunks to the store."""
        if len(embeddings) != len(chunks):
            raise VectorStoreError("Number of embeddings must match number of chunks")

        try:
            # Normalize if needed
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                embeddings = embeddings / norms

            # Train IVF index if needed
            if not self.is_trained:
                if len(embeddings) >= 100:  # Need minimum samples for training
                    self.index.train(embeddings.astype(np.float32))
                    self.is_trained = True
                else:
                    # Not enough samples yet, defer training
                    pass

            # Add to index if trained
            if self.is_trained:
                self.index.add(embeddings.astype(np.float32))

                # Store metadata
                for chunk in chunks:
                    self.chunks[self.next_id] = chunk
                    self.chunk_id_to_index[chunk.id] = self.next_id
                    self.next_id += 1

        except Exception as e:
            raise VectorStoreError(f"Failed to add to FAISS index: {e}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar vectors."""
        if not self.is_trained or self.index.ntotal == 0:
            return []

        try:
            # Normalize query if needed
            if self.normalize:
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm

            # Search
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

            # Collect results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self.chunks:
                    chunk = self.chunks[idx]
                    
                    # Apply filters if provided
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if key == "doc_id" and chunk.doc_id != value:
                                match = False
                                break
                            elif key == "section_path" and value not in chunk.section_path:
                                match = False
                                break
                            elif key in chunk.metadata and chunk.metadata[key] != value:
                                match = False
                                break
                        if not match:
                            continue

                    # Convert distance to similarity score
                    if self.metric == "cosine":
                        score = float(dist)  # Inner product is similarity for normalized vectors
                    else:
                        score = 1.0 / (1.0 + float(dist))  # Convert distance to similarity

                    results.append((chunk, score))

            return results[:k]

        except Exception as e:
            raise VectorStoreError(f"Failed to search FAISS index: {e}")

    def delete(self, chunk_ids: List[str]) -> None:
        """Delete vectors by chunk IDs."""
        try:
            indices_to_remove = []
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_id_to_index:
                    idx = self.chunk_id_to_index[chunk_id]
                    indices_to_remove.append(idx)
                    del self.chunks[idx]
                    del self.chunk_id_to_index[chunk_id]

            if indices_to_remove:
                # FAISS doesn't support direct deletion, need to rebuild
                # This is a simplified approach - in production, use IDMap2
                pass

        except Exception as e:
            raise VectorStoreError(f"Failed to delete from FAISS index: {e}")

    def save(self, path: Optional[str] = None) -> None:
        """Save the vector store to disk."""
        path = path or self.index_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save index
            faiss.write_index(self.index, f"{path}.index")

            # Save metadata
            metadata = {
                "chunks": {k: self._chunk_to_dict(v) for k, v in self.chunks.items()},
                "chunk_id_to_index": self.chunk_id_to_index,
                "next_id": self.next_id,
                "is_trained": self.is_trained,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
            }
            with open(f"{path}.meta", "wb") as f:
                pickle.dump(metadata, f)

        except Exception as e:
            raise VectorStoreError(f"Failed to save FAISS index: {e}")

    def load(self, path: Optional[str] = None) -> None:
        """Load the vector store from disk."""
        path = path or self.index_path

        try:
            # Load index
            if os.path.exists(f"{path}.index"):
                self.index = faiss.read_index(f"{path}.index")

                # Load metadata
                if os.path.exists(f"{path}.meta"):
                    with open(f"{path}.meta", "rb") as f:
                        metadata = pickle.load(f)
                        self.chunks = {
                            k: self._dict_to_chunk(v) for k, v in metadata["chunks"].items()
                        }
                        self.chunk_id_to_index = metadata["chunk_id_to_index"]
                        self.next_id = metadata["next_id"]
                        self.is_trained = metadata["is_trained"]

        except Exception as e:
            raise VectorStoreError(f"Failed to load FAISS index: {e}")

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.index.reset()
        self.chunks.clear()
        self.chunk_id_to_index.clear()
        self.next_id = 0
        self.is_trained = self.index_type != "IVF"

    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": chunk.id,
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "section_path": chunk.section_path,
            "metadata": chunk.metadata,
            "token_count": chunk.token_count,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "prev_chunk_id": chunk.prev_chunk_id,
            "next_chunk_id": chunk.next_chunk_id,
        }

    def _dict_to_chunk(self, data: Dict[str, Any]) -> Chunk:
        """Convert dictionary to chunk."""
        return Chunk(
            id=data["id"],
            text=data["text"],
            doc_id=data["doc_id"],
            section_path=data["section_path"],
            metadata=data["metadata"],
            token_count=data["token_count"],
            start_page=data.get("start_page"),
            end_page=data.get("end_page"),
            prev_chunk_id=data.get("prev_chunk_id"),
            next_chunk_id=data.get("next_chunk_id"),
        )

    @property
    def size(self) -> int:
        """Get number of vectors in the store."""
        return self.index.ntotal if self.is_trained else 0


class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create(
        store_type: str,
        dimension: int,
        **kwargs,
    ) -> VectorStore:
        """Create a vector store instance."""
        if store_type == "faiss":
            return FAISSVectorStore(
                dimension=dimension,
                index_type=kwargs.get("index_type", "Flat"),
                metric=kwargs.get("metric", "cosine"),
                index_path=kwargs.get("index_path"),
            )
        # Add pgvector implementation here when needed
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")


def get_vector_store(dimension: int) -> VectorStore:
    """Get vector store instance based on configuration."""
    return VectorStoreFactory.create(
        store_type=settings.vector_db_type,
        dimension=dimension,
    )