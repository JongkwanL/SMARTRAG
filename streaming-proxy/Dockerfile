# Multi-stage Dockerfile for SmartRAG Streaming Proxy

# Build stage
FROM golang:1.21-alpine AS builder

# Install dependencies
RUN apk add --no-cache git ca-certificates

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o streaming-proxy ./cmd

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates curl
WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/streaming-proxy .

# Expose HTTP port and metrics port
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables with defaults
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV BACKEND_BASE_URL=http://localhost:8000
ENV METRICS_ENABLED=true
ENV METRICS_PORT=9090
ENV STREAMING_MAX_CLIENTS=1000
ENV WS_ENABLE_COMPRESSION=true
ENV SSE_ENABLE_CORS=true

# Run the streaming proxy
CMD ["./streaming-proxy"]