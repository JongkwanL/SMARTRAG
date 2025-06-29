#!/bin/bash

# Script to generate Go code from protobuf definitions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Generating protobuf code for vector search service...${NC}"

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}Error: protoc is not installed. Please install Protocol Buffers compiler.${NC}"
    echo "Installation instructions:"
    echo "  macOS: brew install protobuf"
    echo "  Ubuntu: apt-get install protobuf-compiler"
    exit 1
fi

# Check if protoc-gen-go is installed
if ! command -v protoc-gen-go &> /dev/null; then
    echo -e "${YELLOW}Warning: protoc-gen-go not found. Installing...${NC}"
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

# Check if protoc-gen-go-grpc is installed
if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo -e "${YELLOW}Warning: protoc-gen-go-grpc not found. Installing...${NC}"
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create output directory if it doesn't exist
mkdir -p pkg/proto

# Generate Go code from protobuf
echo -e "${GREEN}Generating Go code from vector_search.proto...${NC}"

protoc \
    --proto_path=pkg/proto \
    --go_out=pkg/proto \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/proto \
    --go-grpc_opt=paths=source_relative \
    pkg/proto/vector_search.proto

if [ $? -eq 0 ]; then
    echo -e "${GREEN} Protocol buffer compilation successful!${NC}"
    echo "Generated files:"
    ls -la pkg/proto/*.pb.go 2>/dev/null || echo "  (Generated files will appear after first compilation)"
else
    echo -e "${RED} Protocol buffer compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Done! You can now build the vector search server.${NC}"