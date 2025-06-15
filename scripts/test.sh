#!/bin/bash

# SmartRAG test runner script
set -e

echo "🧪 Running SmartRAG tests..."

# Load environment variables
if [ -f .env.test ]; then
    echo "📄 Loading test environment variables..."
    export $(cat .env.test | grep -v '^#' | xargs)
elif [ -f .env ]; then
    echo "📄 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Install test dependencies
echo "📥 Installing test dependencies..."
pip install pytest pytest-asyncio pytest-cov httpx

# Run linting
echo "🔍 Running code linting..."
if command -v black &> /dev/null; then
    black --check src/ tests/
fi

if command -v isort &> /dev/null; then
    isort --check-only src/ tests/
fi

if command -v flake8 &> /dev/null; then
    flake8 src/ tests/
fi

# Run type checking
echo "🔬 Running type checking..."
if command -v mypy &> /dev/null; then
    mypy src/
fi

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

echo "✅ All tests completed!"