# Multi-stage Dockerfile for SmartRAG Vector Search Server

# Build stage
FROM golang:1.21-alpine AS builder

# Install dependencies for building
RUN apk add --no-cache git ca-certificates protobuf protobuf-dev

# Set working directory
WORKDIR /app

# Install protoc-gen-go and protoc-gen-go-grpc
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Generate protobuf code
RUN ./scripts/generate_proto.sh

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o vector-search ./cmd/server

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates curl
WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/vector-search .

# Create data directory for index storage
RUN mkdir -p ./data

# Expose gRPC port and metrics port
EXPOSE 50051 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Environment variables with defaults
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=50051
ENV VECTOR_DIMENSION=384
ENV VECTOR_INDEX_TYPE=hnsw
ENV VECTOR_INDEX_PATH=./data/vector.index
ENV METRICS_ENABLED=true
ENV METRICS_PORT=9090

# Run the vector search server
CMD ["./vector-search"]