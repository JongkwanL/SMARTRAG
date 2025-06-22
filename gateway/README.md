# SmartRAG API Gateway

High-performance Go-based API Gateway for the SmartRAG system, providing authentication, rate limiting, caching, and request proxying to the Python backend.

## Features

- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Redis-based rate limiting with configurable limits
- **Response Caching**: Intelligent HTTP response caching with Redis
- **Request Proxying**: High-performance proxy to Python FastAPI backend
- **Metrics**: Prometheus metrics collection
- **CORS Support**: Configurable CORS middleware
- **Health Checks**: Built-in health monitoring
- **Graceful Shutdown**: Proper server lifecycle management

## Architecture

```
Client -> API Gateway (Go) -> SmartRAG Backend (Python FastAPI)
             |
             v
          Redis (Cache + Rate Limiting)
```

## Configuration

Environment variables:

```bash
# Server
PORT=8080
READ_TIMEOUT=30
WRITE_TIMEOUT=30
IDLE_TIMEOUT=120
ENVIRONMENT=development

# Backend
BACKEND_URL=http://localhost:8000
BACKEND_MAX_CONNS=100
BACKEND_MAX_IDLE_CONNS=10
BACKEND_CONN_TIMEOUT=10
BACKEND_RESPONSE_TIMEOUT=30
BACKEND_MAX_RETRIES=3

# Redis
REDIS_ADDRESS=localhost:6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_POOL_SIZE=10

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_TOKEN_TTL=3600

# Rate Limiting
RATE_LIMIT_RPM=100
RATE_LIMIT_BURST=20
RATE_LIMIT_WINDOW=60

# CORS
CORS_ALLOWED_ORIGINS=*
CORS_ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOWED_HEADERS=*
```

## API Endpoints

### Public Endpoints
- `GET /health` - Gateway health check
- `GET /metrics` - Prometheus metrics

### Protected Endpoints (require JWT)
- `POST /api/v1/ingest` - Document ingestion (cached 5min)
- `POST /api/v1/query` - Query processing (cached 1min)
- `ANY /api/v1/*` - All other endpoints proxied to backend

## Running

### Development
```bash
go run main.go
```

### Docker
```bash
docker build -t smartrag-gateway .
docker run -p 8080:8080 smartrag-gateway
```

### Production
```bash
go build -o gateway
./gateway
```

## Dependencies

- Gin (HTTP framework)
- Redis (caching and rate limiting)
- JWT (authentication)
- Prometheus (metrics)
- Zap (logging)

## Performance

- Sub-millisecond request routing
- Redis-backed caching for faster responses
- Connection pooling for backend communication
- Configurable timeouts and limits
- Prometheus metrics for monitoring