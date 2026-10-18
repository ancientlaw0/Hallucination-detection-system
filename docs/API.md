# API Reference

Base URL: `http://localhost:8000`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## GET /

Health ping. Returns a simple alive message.

**Response 200**
```json
{ "message": "Hallucination Detection API is running" }
```

---

## GET /health

Detailed health check. Confirms the model is loaded and reports the compute device.

**Response 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if the server is running |
| `model_loaded` | bool | Always `true` — model is loaded at startup |
| `device` | string | `"cpu"` or `"cuda"` |

---

## GET /model-info

Returns metadata about the loaded model.

**Response 200**
```json
{
  "model": "Fine-tuned DeBERTa",
  "task": "response-level hallucination detection",
  "threshold": 0.4997,
  "device": "cpu"
}
```

---

## POST /verify

**Core endpoint.** Classify whether an AI-generated response contains hallucinations
relative to the provided evidence.

**Request body** (`application/json`)
```json
{
  "evidence": "The Eiffel Tower is in Paris, France, built in 1889.",
  "response": "The Eiffel Tower is in London and was built in 1920."
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `evidence` | string | 1–100,000 chars | The reference text (ground truth) |
| `response` | string | 1–50,000 chars | The AI-generated text to verify |

**Response 200**
```json
{
  "prediction": "potential_hallucination",
  "hallucination_probability": 0.8734,
  "threshold": 0.4997,
  "warning": "This is a response-level prediction, not proof of factual incorrectness. Exact faulty claims are not identified."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"potential_hallucination"` or `"no_annotated_hallucination_detected"` |
| `hallucination_probability` | float | Probability [0, 1] that the response contains a hallucination |
| `threshold` | float | The decision boundary used |
| `warning` | string | Reminder that this is a response-level signal |

**Errors**

| Status | Cause |
|--------|-------|
| 422 | Missing field, empty string, or text exceeds max length |
| 500 | Internal inference error (check server logs) |

---

## POST /documents/upload

Upload a PDF document. Text is extracted, chunked, embedded with
`sentence-transformers/all-MiniLM-L6-v2`, and indexed in an in-memory FAISS store.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | A PDF file (≤ 20 MB, text-based) |

**Response 200**
```json
{
  "document_id": "a1b2c3d4-...",
  "filename": "a1b2c3d4-....pdf",
  "pages_indexed": 12,
  "chunks_indexed": 48,
  "original_filename": "my_paper.pdf"
}
```

**Errors**

| Status | Cause |
|--------|-------|
| 400 | Not a PDF, empty file, exceeds 20 MB, or no extractable text (scanned PDF) |
| 422 | No file attached |

---

## POST /retrieve

Semantic search over all uploaded documents.

**Request body** (`application/json`)
```json
{
  "query": "What is the capital of France?",
  "top_k": 3
}
```

| Field | Type | Constraints | Default |
|-------|------|-------------|---------|
| `query` | string | 1–5,000 chars | — |
| `top_k` | int | 1–10 | 5 |

**Response 200**
```json
{
  "query": "What is the capital of France?",
  "count": 3,
  "results": [
    {
      "document_id": "a1b2c3d4-...",
      "filename": "a1b2c3d4-....pdf",
      "page": 1,
      "text": "Paris is the capital and most populous city of France...",
      "similarity": 0.8921
    }
  ]
}
```

Each result includes a `similarity` score (cosine similarity, range 0–1).
Higher is more relevant.
