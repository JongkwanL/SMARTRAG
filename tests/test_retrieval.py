"""Tests for retrieval modules."""

import numpy as np
import pytest

from src.core.document_processor import Chunk
from src.retrieval.vector_store import FAISSVectorStore


class TestFAISSVectorStore:
    """Test FAISS vector store."""

    def test_init_flat_index(self):
        """Test initialization with flat index."""
        store = FAISSVectorStore(
            dimension=384,
            index_type="Flat",
            metric="cosine",
        )
        assert store.dimension == 384
        assert store.index_type == "Flat"
        assert store.metric == "cosine"
        assert store.normalize is True

    def test_init_euclidean_index(self):
        """Test initialization with Euclidean metric."""
        store = FAISSVectorStore(
            dimension=384,
            index_type="Flat",
            metric="euclidean",
        )
        assert store.normalize is False

    def test_add_embeddings(self, vector_store, sample_chunks):
        """Test adding embeddings to store."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        
        # Initially empty
        assert vector_store.size == 0
        
        # Add embeddings
        vector_store.add(embeddings, sample_chunks)
        
        # Should now have embeddings
        assert vector_store.size == len(sample_chunks)

    def test_search_embeddings(self, vector_store, sample_chunks):
        """Test searching embeddings."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        vector_store.add(embeddings, sample_chunks)
        
        # Search with similar query
        query_embedding = embeddings[0] + np.random.normal(0, 0.1, 384)
        results = vector_store.search(query_embedding, k=2)
        
        assert len(results) <= 2
        if results:
            chunk, score = results[0]
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    def test_search_with_filters(self, vector_store, sample_chunks):
        """Test searching with filters."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        vector_store.add(embeddings, sample_chunks)
        
        query_embedding = embeddings[0]
        
        # Filter by doc_id
        results = vector_store.search(
            query_embedding,
            k=5,
            filters={"doc_id": "doc_1"}
        )
        
        for chunk, score in results:
            assert chunk.doc_id == "doc_1"

    def test_search_empty_store(self, vector_store):
        """Test searching empty store."""
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_embedding, k=5)
        assert len(results) == 0

    def test_save_and_load(self, vector_store, sample_chunks, temp_dir):
        """Test saving and loading vector store."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        vector_store.add(embeddings, sample_chunks)
        
        # Save
        save_path = f"{temp_dir}/test_index"
        vector_store.save(save_path)
        
        # Create new store and load
        new_store = FAISSVectorStore(
            dimension=384,
            index_type="Flat",
            metric="cosine",
        )
        new_store.load(save_path)
        
        # Should have same data
        assert new_store.size == vector_store.size
        
        # Test search gives similar results
        query_embedding = embeddings[0]
        original_results = vector_store.search(query_embedding, k=2)
        loaded_results = new_store.search(query_embedding, k=2)
        
        assert len(original_results) == len(loaded_results)

    def test_clear_store(self, vector_store, sample_chunks):
        """Test clearing vector store."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        vector_store.add(embeddings, sample_chunks)
        
        assert vector_store.size > 0
        
        vector_store.clear()
        assert vector_store.size == 0

    def test_delete_chunks(self, vector_store, sample_chunks):
        """Test deleting chunks by ID."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        vector_store.add(embeddings, sample_chunks)
        
        # Delete some chunks
        chunk_ids = [sample_chunks[0].id]
        vector_store.delete(chunk_ids)
        
        # Note: FAISS doesn't support direct deletion easily
        # This test verifies the method doesn't crash


class TestSearchEngine:
    """Test search engine functionality."""

    @pytest.mark.asyncio
    async def test_vector_search(self, embedding_generator, vector_store, sample_chunks):
        """Test pure vector search."""
        from src.retrieval.search import SearchEngine
        
        # Generate embeddings for chunks
        texts = [chunk.text for chunk in sample_chunks]
        embeddings = await embedding_generator.generate_embeddings_async(texts)
        
        # Add to vector store
        vector_store.add(embeddings, sample_chunks)
        
        # Create search engine
        search_engine = SearchEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
        )
        
        # Search
        results = await search_engine.search("RAG systems", max_results=2)
        
        assert len(results) <= 2
        if results:
            assert hasattr(results[0], 'text')
            assert hasattr(results[0], 'metadata')

    @pytest.mark.asyncio
    async def test_hybrid_search(self, embedding_generator, vector_store, sample_chunks):
        """Test hybrid vector + BM25 search."""
        from src.retrieval.search import SearchEngine
        
        # Generate embeddings
        texts = [chunk.text for chunk in sample_chunks]
        embeddings = await embedding_generator.generate_embeddings_async(texts)
        vector_store.add(embeddings, sample_chunks)
        
        # Create search engine with BM25
        search_engine = SearchEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            use_bm25=True,
        )
        
        # Add documents to BM25
        for chunk in sample_chunks:
            search_engine.add_to_bm25(chunk)
        
        # Search
        results = await search_engine.search("RAG vector embeddings", max_results=3)
        
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_mmr_reranking(self, embedding_generator, vector_store, sample_chunks):
        """Test MMR reranking for diversity."""
        from src.retrieval.search import SearchEngine
        
        # Generate embeddings
        texts = [chunk.text for chunk in sample_chunks]
        embeddings = await embedding_generator.generate_embeddings_async(texts)
        vector_store.add(embeddings, sample_chunks)
        
        # Create search engine
        search_engine = SearchEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            use_mmr=True,
            mmr_lambda=0.7,
        )
        
        # Search
        results = await search_engine.search("implementation details", max_results=2)
        
        assert len(results) <= 2

    def test_query_expansion(self):
        """Test query expansion functionality."""
        from src.retrieval.search import SearchEngine
        
        search_engine = SearchEngine(
            vector_store=None,
            embedding_generator=None,
        )
        
        # Test synonym expansion
        expanded = search_engine._expand_query("ML model")
        assert "machine learning" in expanded.lower()
        
        # Test technical term expansion  
        expanded = search_engine._expand_query("API")
        assert "application programming interface" in expanded.lower()


class TestSimilarityMetrics:
    """Test similarity computation."""

    def test_cosine_similarity(self, embedding_generator):
        """Test cosine similarity computation."""
        # Create test embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        emb3 = np.array([1.0, 0.0, 0.0])
        
        # Test similarity
        sim_different = embedding_generator.compute_similarity(emb1, emb2, "cosine")
        sim_same = embedding_generator.compute_similarity(emb1, emb3, "cosine")
        
        assert sim_different == 0.0  # Orthogonal vectors
        assert sim_same == 1.0      # Identical vectors

    def test_batch_similarities(self, embedding_generator):
        """Test batch similarity computation."""
        query_emb = np.array([1.0, 0.0, 0.0])
        doc_embs = np.array([
            [1.0, 0.0, 0.0],  # Same as query
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.5, 0.5, 0.0],  # Partial overlap
        ])
        
        similarities = embedding_generator.compute_similarities(
            query_emb, doc_embs, "cosine"
        )
        
        assert len(similarities) == 3
        assert similarities[0] == 1.0  # Perfect match
        assert similarities[1] == 0.0  # No match
        assert 0.0 < similarities[2] < 1.0  # Partial match


class TestRetrievalMetrics:
    """Test retrieval evaluation metrics."""

    def test_recall_at_k(self):
        """Test recall@k metric."""
        from src.retrieval.search import compute_recall_at_k
        
        # Mock relevant and retrieved documents
        relevant_docs = {"doc1", "doc2", "doc3"}
        retrieved_docs = ["doc1", "doc4", "doc2"]
        
        recall = compute_recall_at_k(relevant_docs, retrieved_docs, k=3)
        assert recall == 2/3  # 2 relevant out of 3 total relevant

    def test_precision_at_k(self):
        """Test precision@k metric."""
        from src.retrieval.search import compute_precision_at_k
        
        relevant_docs = {"doc1", "doc2", "doc3"}
        retrieved_docs = ["doc1", "doc4", "doc2"]
        
        precision = compute_precision_at_k(relevant_docs, retrieved_docs, k=3)
        assert precision == 2/3  # 2 relevant out of 3 retrieved

    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        from src.retrieval.search import compute_mrr
        
        # Each query has a list of retrieved docs and set of relevant docs
        queries = [
            (["doc1", "doc2", "doc3"], {"doc2"}),  # Relevant at position 2
            (["doc4", "doc1", "doc5"], {"doc1"}),  # Relevant at position 2
            (["doc6", "doc7", "doc8"], {"doc9"}),  # No relevant docs
        ]
        
        mrr = compute_mrr(queries)
        expected_mrr = (1/2 + 1/2 + 0) / 3
        assert abs(mrr - expected_mrr) < 1e-6