# Multi-stage Dockerfile for SmartRAG Monitoring Agent

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
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-extldflags "-static"' -o monitoring-agent ./cmd

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates curl
WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/monitoring-agent .

# Copy configuration template
COPY --from=builder /app/config.example.yaml ./config.yaml

# Create directories for data
RUN mkdir -p /data /logs

# Expose HTTP port and metrics port
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables with defaults
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV METRICS_PORT=9090
ENV MONITORING_ENABLED=true
ENV MONITORING_COLLECTION_INTERVAL=30s
ENV MONITORING_HEALTH_CHECK_INTERVAL=60s
ENV LOGGING_LEVEL=info
ENV STORAGE_TYPE=memory

# Run the monitoring agent
CMD ["./monitoring-agent"]