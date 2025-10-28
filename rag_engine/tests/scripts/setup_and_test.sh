#!/bin/bash
# Test setup script for RAG engine (WSL/Linux)
# Run from project root: bash rag_engine/tests/scripts/setup_and_test.sh

set -e  # Exit on error

echo "=== RAG Engine Test Setup Script ==="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version || { echo "Error: python3 not found"; exit 1; }

# Create venv if it doesn't exist
if [ ! -d ".venv-wsl" ]; then
    echo "2. Creating virtual environment..."
    python3 -m venv .venv-wsl
else
    echo "2. Virtual environment exists, reusing it"
fi

# Activate venv
echo "3. Activating virtual environment..."
source .venv-wsl/bin/activate

# Upgrade pip
echo "4. Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "5. Installing dependencies from requirements.txt..."
pip install -r rag_engine/requirements.txt --quiet

echo ""
echo "=== Dependencies installed! ==="
echo ""

# Run linter
echo "6. Running linter (Ruff)..."
ruff check rag_engine/ --fix || echo "Some linter issues found (non-critical)"

echo ""

# Run tests
echo "7. Running tests..."
python -m pytest rag_engine/tests/ -v --tb=short

echo ""
echo "=== Test setup and execution complete! ==="
echo ""
echo "To activate the environment manually:"
echo "  source .venv-wsl/bin/activate"
echo ""
echo "To run tests only:"
echo "  python -m pytest rag_engine/tests/ -v"

