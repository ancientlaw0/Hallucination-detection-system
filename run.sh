#!/usr/bin/env bash
# ==============================================================================
# run.sh — One-click launcher for the Hallucination Detection System
#
# This script:
#   1. Detects or creates the Python virtual environment (.venv)
#   2. Installs / verifies required packages
#   3. Verifies that the fine-tuned model artifacts exist
#   4. Starts the API server with the interactive web UI
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "   🚀 Starting Hallucination Detection System"
echo "======================================================================"

# 1. Check / Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activate Virtual Environment
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Check and Install Requirements
if ! python -c "import fastapi, torch, transformers, sentence_transformers, faiss, pymupdf" >/dev/null 2>&1; then
    echo "📥 Installing required dependencies from requirements.txt..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    fi
    echo "✓ Dependencies installed."
fi

# 4. Check Model Artifacts
MODEL_PATH="$PROJECT_DIR/artifacts/fine_tuned_hallucination_verifier"
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/model.safetensors" ]; then
    echo "⚠️  WARNING: Model not found at $MODEL_PATH"
    echo "   Please place the model in artifacts/fine_tuned_hallucination_verifier"
else
    echo "✓ Fine-tuned DeBERTa model verified at $MODEL_PATH"
fi

# 5. Set Environment Safety Settings (OpenMP collision avoidance on macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

echo ""
echo "======================================================================"
echo "   🟢 Server starting on http://localhost:8000"
echo "   👉 Testing Web UI:     http://localhost:8000/ui"
echo "   👉 Interactive Docs:   http://localhost:8000/docs"
echo "   👉 Health Status:      http://localhost:8000/health"
echo "======================================================================"
echo "Press Ctrl+C to stop the server."
echo ""

# 6. Launch Application
exec python run.py
