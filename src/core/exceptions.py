"""Custom exceptions for SmartRAG."""


class SmartRAGException(Exception):
    """Base exception for SmartRAG."""

    pass


class DocumentProcessingError(SmartRAGException):
    """Exception raised during document processing."""

    pass


class EmbeddingGenerationError(SmartRAGException):
    """Exception raised during embedding generation."""

    pass


class VectorStoreError(SmartRAGException):
    """Exception raised during vector store operations."""

    pass


class LLMError(SmartRAGException):
    """Exception raised during LLM operations."""

    pass


class CacheError(SmartRAGException):
    """Exception raised during cache operations."""

    pass


class ConfigurationError(SmartRAGException):
    """Exception raised for configuration errors."""

    pass


class ValidationError(SmartRAGException):
    """Exception raised for validation errors."""

    pass


class RateLimitError(SmartRAGException):
    """Exception raised when rate limit is exceeded."""

    pass


class AuthenticationError(SmartRAGException):
    """Exception raised for authentication errors."""

    pass


class StorageError(SmartRAGException):
    """Exception raised during storage operations."""

    pass