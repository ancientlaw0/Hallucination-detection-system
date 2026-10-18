# Contributing Guide

## Getting started

```bash
git clone <repo-url>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Place the model in `artifacts/fine_tuned_hallucination_verifier/` (see `docs/README.md`).

---

## Running tests

```bash
pytest tests/ -v
```

The test suite mocks the transformer model, so no GPU or model download is needed.
To run a single test file:

```bash
pytest tests/test_verify.py -v
```

---

## Adding a new endpoint

1. Add the route in `Backend/main.py`.
2. Add a Pydantic request/response model if needed.
3. Write tests in `tests/test_<feature>.py`.
4. Document the endpoint in `docs/API.md`.

---

## Code style

- Python 3.11+
- Type hints on all function signatures.
- Docstrings on all public functions and classes.
- Keep routes thin — move business logic into helper modules.

---

## Directory conventions

| Directory | Contents |
|-----------|---------|
| `Backend/` | All FastAPI application code |
| `artifacts/` | Model weights — never commit to Git |
| `data/uploads/` | Uploaded PDFs at runtime — not committed |
| `tests/` | All test files, one per feature area |
| `docs/` | All documentation |
