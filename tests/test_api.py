"""Tests for API endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, test_client: TestClient):
        """Test health check returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestIngestEndpoint:
    """Test document ingestion endpoint."""

    @patch("src.api.endpoints.get_document_pipeline")
    @patch("src.api.endpoints.get_embedding_generator")
    @patch("src.api.endpoints.get_vector_store")
    def test_ingest_document_success(
        self,
        mock_vector_store,
        mock_embedding_gen,
        mock_pipeline,
        test_client: TestClient,
        sample_text: str,
    ):
        """Test successful document ingestion."""
        # Mock pipeline
        mock_chunks = [
            MagicMock(id="chunk_1", text="Sample chunk 1"),
            MagicMock(id="chunk_2", text="Sample chunk 2"),
        ]
        mock_pipeline.return_value.process_document = AsyncMock(
            return_value=(mock_chunks, {"total_chunks": 2})
        )

        # Mock embedding generator
        mock_embedding_gen.return_value.generate_embeddings_async = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        )

        # Mock vector store
        mock_vector_store.return_value.add = MagicMock()

        response = test_client.post(
            "/ingest",
            json={
                "content": sample_text,
                "doc_id": "test_doc",
                "doc_type": "md",
                "metadata": {"source": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["doc_id"] == "test_doc"
        assert data["stats"]["total_chunks"] == 2

    def test_ingest_missing_content(self, test_client: TestClient):
        """Test ingestion with missing content."""
        response = test_client.post(
            "/ingest",
            json={"doc_id": "test_doc"},
        )
        assert response.status_code == 422  # Validation error

    def test_ingest_missing_doc_id(self, test_client: TestClient):
        """Test ingestion with missing doc_id."""
        response = test_client.post(
            "/ingest",
            json={"content": "test content"},
        )
        assert response.status_code == 422  # Validation error


class TestQueryEndpoint:
    """Test query endpoint."""

    @patch("src.api.endpoints.get_search_engine")
    @patch("src.api.endpoints.get_llm_client")
    def test_query_success(
        self,
        mock_llm_client,
        mock_search_engine,
        test_client: TestClient,
    ):
        """Test successful query."""
        # Mock search results
        mock_search_results = [
            MagicMock(
                text="Relevant context about RAG",
                metadata={"doc_id": "doc_1", "score": 0.9},
            )
        ]
        mock_search_engine.return_value.search = AsyncMock(
            return_value=mock_search_results
        )

        # Mock LLM response
        mock_llm_client.return_value.generate_response = AsyncMock(
            return_value={
                "response": "RAG is a powerful technique...",
                "usage": {"tokens": 50},
            }
        )

        response = test_client.post(
            "/query",
            json={
                "query": "What is RAG?",
                "max_chunks": 5,
                "include_sources": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert "usage" in data

    def test_query_missing_query(self, test_client: TestClient):
        """Test query with missing query parameter."""
        response = test_client.post("/query", json={})
        assert response.status_code == 422  # Validation error

    @patch("src.api.endpoints.get_search_engine")
    @patch("src.api.endpoints.get_llm_client")
    def test_query_streaming(
        self,
        mock_llm_client,
        mock_search_engine,
        test_client: TestClient,
    ):
        """Test streaming query response."""
        # Mock search results
        mock_search_engine.return_value.search = AsyncMock(
            return_value=[MagicMock(text="Context")]
        )

        # Mock streaming response
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_llm_client.return_value.stream_response = AsyncMock(
            return_value=mock_stream()
        )

        response = test_client.post(
            "/query",
            json={
                "query": "What is RAG?",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_endpoint(self, test_client: TestClient):
        """Test metrics endpoint returns Prometheus format."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for some expected metrics
        content = response.text
        assert "smartrag_" in content


class TestRateLimiting:
    """Test rate limiting functionality."""

    @patch("src.api.middleware.get_redis_client")
    def test_rate_limit_not_exceeded(
        self,
        mock_redis,
        test_client: TestClient,
    ):
        """Test request under rate limit."""
        # Mock Redis to return low count
        mock_redis.return_value.get = AsyncMock(return_value="5")
        mock_redis.return_value.setex = AsyncMock()

        response = test_client.get("/health")
        assert response.status_code == 200

    @patch("src.api.middleware.get_redis_client")
    def test_rate_limit_exceeded(
        self,
        mock_redis,
        test_client: TestClient,
    ):
        """Test request over rate limit."""
        # Mock Redis to return high count
        mock_redis.return_value.get = AsyncMock(return_value="101")

        response = test_client.get("/health")
        assert response.status_code == 429  # Too Many Requests
        assert "Rate limit exceeded" in response.json()["detail"]


class TestAuthentication:
    """Test authentication middleware."""

    def test_public_endpoint_no_auth(self, test_client: TestClient):
        """Test public endpoints don't require auth."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_protected_endpoint_no_token(self, test_client: TestClient):
        """Test protected endpoint without token."""
        response = test_client.post("/ingest", json={"content": "test"})
        # Note: This depends on whether ingest is protected
        # Adjust based on actual implementation

    def test_protected_endpoint_valid_token(self, test_client: TestClient):
        """Test protected endpoint with valid token."""
        # This would require a valid JWT token
        # Implementation depends on auth strategy
        pass

    def test_protected_endpoint_invalid_token(self, test_client: TestClient):
        """Test protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = test_client.post(
            "/ingest",
            json={"content": "test"},
            headers=headers,
        )
        # Check for 401 Unauthorized if auth is implemented
        pass


class TestErrorHandling:
    """Test error handling."""

    @patch("src.api.endpoints.get_document_pipeline")
    def test_internal_server_error(
        self,
        mock_pipeline,
        test_client: TestClient,
    ):
        """Test internal server error handling."""
        # Mock pipeline to raise exception
        mock_pipeline.return_value.process_document = AsyncMock(
            side_effect=Exception("Test error")
        )

        response = test_client.post(
            "/ingest",
            json={
                "content": "test content",
                "doc_id": "test_doc",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data


class TestCORS:
    """Test CORS middleware."""

    def test_cors_headers(self, test_client: TestClient):
        """Test CORS headers are present."""
        response = test_client.options("/health")
        assert response.status_code == 200
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers