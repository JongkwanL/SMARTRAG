"""Configuration management module for SmartRAG."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="SmartRAG", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS",
    )

    # Security
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Database
    database_url: str = Field(env="DATABASE_URL")

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # Vector Database
    vector_db_type: str = Field(default="faiss", env="VECTOR_DB_TYPE")
    faiss_index_path: str = Field(default="/data/faiss_index", env="FAISS_INDEX_PATH")
    pgvector_dim: int = Field(default=768, env="PGVECTOR_DIM")

    # Embedding Model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
    )
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    max_seq_length: int = Field(default=512, env="MAX_SEQ_LENGTH")

    # LLM Settings
    llm_model: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        env="LLM_MODEL",
    )
    llm_api_url: str = Field(default="http://localhost:8001/v1", env="LLM_API_URL")
    llm_api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, env="LLM_TOP_P")
    llm_stream: bool = Field(default=True, env="LLM_STREAM")

    # vLLM Server
    vllm_host: str = Field(default="localhost", env="VLLM_HOST")
    vllm_port: int = Field(default=8001, env="VLLM_PORT")
    vllm_model_path: str = Field(
        default="/models/llama-2-7b-chat-hf",
        env="VLLM_MODEL_PATH",
    )
    vllm_gpu_memory_utilization: float = Field(
        default=0.9,
        env="VLLM_GPU_MEMORY_UTILIZATION",
    )
    vllm_max_model_len: int = Field(default=4096, env="VLLM_MAX_MODEL_LEN")

    # Document Processing
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_document_size_mb: int = Field(default=100, env="MAX_DOCUMENT_SIZE_MB")
    supported_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "html", "docx"],
        env="SUPPORTED_FILE_TYPES",
    )

    # Storage
    s3_endpoint: str = Field(default="http://localhost:9000", env="S3_ENDPOINT")
    s3_access_key: str = Field(default="minioadmin", env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin", env="S3_SECRET_KEY")
    s3_bucket_name: str = Field(default="smartrag-documents", env="S3_BUCKET_NAME")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    use_ssl: bool = Field(default=False, env="USE_SSL")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        env="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_service_name: str = Field(default="smartrag-api", env="OTEL_SERVICE_NAME")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")

    # Performance
    worker_count: int = Field(default=4, env="WORKER_COUNT")
    thread_pool_size: int = Field(default=10, env="THREAD_POOL_SIZE")
    connection_pool_size: int = Field(default=20, env="CONNECTION_POOL_SIZE")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    keep_alive_timeout: int = Field(default=65, env="KEEP_ALIVE_TIMEOUT")

    @validator("vector_db_type")
    def validate_vector_db_type(cls, v: str) -> str:
        """Validate vector database type."""
        allowed = ["faiss", "pgvector"]
        if v not in allowed:
            raise ValueError(f"vector_db_type must be one of {allowed}")
        return v

    @validator("embedding_device")
    def validate_embedding_device(cls, v: str) -> str:
        """Validate embedding device."""
        allowed = ["cpu", "cuda"]
        if v not in allowed:
            raise ValueError(f"embedding_device must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()