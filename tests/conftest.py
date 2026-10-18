"""
conftest.py — Shared fixtures for the Hallucination Detection test suite.

Root problem: Backend.main loads the DeBERTa model AND Backend.retrieval loads
sentence-transformers at module-level (top-level code, not inside functions).
Standard `patch()` context managers cannot intercept code that already ran.

Solution: We patch `transformers` and `sentence_transformers` at the SESSION
level BEFORE any Backend module is ever imported.  We use a session-scoped
autouse fixture to hold the patches for the entire test run.
"""

import io
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _make_logits(hallucination_prob: float):
    import torch, math
    h = max(hallucination_prob, 1e-8)
    n = max(1.0 - hallucination_prob, 1e-8)
    return torch.tensor([[math.log(n), math.log(h)]])


def _make_fake_sentence_transformer(dim: int = 8):
    """Return a MagicMock that behaves like a SentenceTransformer."""
    import numpy as np
    st = MagicMock()
    st.get_sentence_embedding_dimension.return_value = dim
    st.get_embedding_dimension.return_value = dim  # new API
    def _encode(texts, **kwargs):
        return np.random.rand(len(texts), dim).astype("float32")
    st.encode.side_effect = _encode
    return st


# --------------------------------------------------------------------------- #
# Session-wide patches — applied BEFORE any Backend module is imported         #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session", autouse=True)
def _patch_heavy_deps(tmp_path_factory):
    """
    Patch transformers + sentence_transformers at session scope so that
    Backend.main and Backend.retrieval can be imported without touching real
    model weights or downloading anything.

    This fixture runs first (autouse=True) and is kept alive for the whole
    session (scope="session").
    """
    import torch

    # ---- Build mocks ----
    model_dir = tmp_path_factory.mktemp("fake_model")
    metrics = {"decision_threshold_selected_on_validation": 0.5}
    (model_dir / "evaluation_metrics.json").write_text(json.dumps(metrics))

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_mdl = MagicMock()
    output = types.SimpleNamespace(logits=_make_logits(0.1))
    mock_mdl.return_value = output
    mock_mdl.to = MagicMock(return_value=mock_mdl)
    mock_mdl.eval = MagicMock(return_value=mock_mdl)

    fake_st = _make_fake_sentence_transformer(dim=8)

    # Clear any previously cached Backend modules so patches take effect.
    for key in [k for k in sys.modules if k.startswith("Backend")]:
        del sys.modules[key]

    with (
        patch.dict(os.environ, {"MODEL_DIR": str(model_dir)}),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok),
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock_mdl),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        yield {
            "model_dir": model_dir,
            "mock_tok": mock_tok,
            "mock_mdl": mock_mdl,
            "fake_st": fake_st,
        }

    # Cleanup after session.
    for key in [k for k in sys.modules if k.startswith("Backend")]:
        del sys.modules[key]


# --------------------------------------------------------------------------- #
# TestClient fixture                                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def client(_patch_heavy_deps):
    """
    FastAPI TestClient.  The heavy deps fixture above has already applied all
    patches, so importing Backend.main here is safe.
    """
    from fastapi.testclient import TestClient
    import Backend.main as main_module

    mock_mdl = _patch_heavy_deps["mock_mdl"]
    mock_tok = _patch_heavy_deps["mock_tok"]

    with TestClient(main_module.app) as c:
        c._mock_tok = mock_tok
        c._mock_mdl = mock_mdl
        yield c


# --------------------------------------------------------------------------- #
# Minimal PDF bytes for upload tests                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def minimal_pdf() -> bytes:
    """
    Minimal valid text-based PDF (one page, one line of text).
    PyMuPDF can extract "Hello world" from this.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello world) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n"
    )
