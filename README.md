# Hallucination Detection System

A research prototype that automatically detects whether an AI-generated response contains
**hallucinations** — claims that are not supported by the provided evidence.

## What it does

Given:
1. A piece of **evidence** (e.g. a paragraph from a PDF or a retrieved document)
2. An AI-generated **response** to check

The system returns a verdict: `potential_hallucination` or `no_annotated_hallucination_detected`,
along with a confidence score.

It also supports uploading PDF documents and retrieving relevant passages via semantic search,
enabling a full **Retrieval-Augmented Generation (RAG)** verification workflow.

---

## Quick Start

### Option A: One-Command Launch (Recommended)

Simply execute `run.sh`:
```bash
./run.sh
```
This automatically configures the virtual environment, verifies dependencies, checks the model, and starts the server with both the Web UI and API.

---

### Option B: Manual Setup

1. **Set up the virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start the server:**
```bash
python run.py
```

---

## Accessing the Interfaces

Once started, the system provides two interactive interfaces:
- 🖥️ **Testing Web UI:** **`http://localhost:8000/ui`** — Minimal vanilla testing dashboard to try out evidence vs. response hallucination scoring and PDF retrieval.
- 📖 **Interactive API Docs:** **`http://localhost:8000/docs`** — OpenAPI / Swagger interface for direct JSON endpoint exploration.
- 🏥 **Health Status:** **`http://localhost:8000/health`**

---

## Project Structure

```
Hallucination-detection-system/
├── artifacts/                          ← model weights (not committed to Git)
│   └── fine_tuned_hallucination_verifier/
├── Backend/
│   ├── __init__.py
│   ├── main.py                         ← FastAPI application & inference logic
│   └── retrieval.py                    ← PDF ingestion + FAISS vector store
├── data/
│   └── uploads/                        ← uploaded PDFs stored here
├── docs/
│   ├── README.md                       ← this file
│   ├── API.md                          ← endpoint reference
│   ├── MODEL.md                        ← model card
│   ├── CONTRIBUTING.md                 ← how to extend the project
│   └── EXPLANATION.md                  ← plain-English explanation for presentations
├── tests/
│   ├── conftest.py                     ← shared fixtures (mocked model)
│   ├── test_health.py
│   ├── test_verify.py
│   ├── test_documents.py
│   └── test_retrieval_unit.py
├── requirements.txt                    ← runtime dependencies
├── requirements-dev.txt                ← + test tools
└── run.py                              ← one-command server launcher
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests mock the transformer model — no GPU required.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `transformers` | Load and run the DeBERTa model |
| `torch` | Deep learning runtime |
| `sentence-transformers` | Embed text for semantic search |
| `faiss-cpu` | Fast approximate nearest-neighbour search |
| `pymupdf` | Extract text from PDF files |
| `safetensors` | Safe model weight format |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `artifacts/fine_tuned_hallucination_verifier` | Path to the model directory |
