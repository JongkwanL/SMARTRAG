"""
API module for SmartRAG.

This module provides FastAPI-based REST API endpoints for the SmartRAG system,
including document processing, search, and RAG functionality.
"""

from .main import app, create_app
from .models import (
    DocumentRequest,
    DocumentResponse,
    SearchRequest,
    SearchResponse,
    RAGRequest,
    RAGResponse,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    "app",
    "create_app",
    "DocumentRequest",
    "DocumentResponse", 
    "SearchRequest",
    "SearchResponse",
    "RAGRequest",
    "RAGResponse",
    "HealthResponse",
    "ErrorResponse",
]