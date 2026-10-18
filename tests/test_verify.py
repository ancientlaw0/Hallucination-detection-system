"""
test_verify.py — Tests for POST /verify (the core hallucination detection endpoint).

Tests cover:
  - Valid requests → correct response shape
  - Probability range is always [0, 1]
  - Warning is always present
  - Threshold is echoed back
  - Empty fields → 422 Validation Error
  - Missing fields → 422 Validation Error
  - Oversized text → 422 Validation Error
"""

import types
import pytest


SAMPLE_EVIDENCE = (
    "The Eiffel Tower is located in Paris, France. "
    "It was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair."
)

SAMPLE_RESPONSE_OK = (
    "The Eiffel Tower stands in Paris and was constructed between 1887 and 1889."
)

SAMPLE_RESPONSE_HALL = (
    "The Eiffel Tower is located in London, England, and was built in 1920."
)


class TestVerifyHappyPath:
    def test_returns_200(self, client):
        response = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        )
        assert response.status_code == 200

    def test_prediction_field_exists(self, client):
        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        assert "prediction" in data

    def test_prediction_is_valid_label(self, client):
        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        valid_labels = {
            "potential_hallucination",
            "no_annotated_hallucination_detected",
        }
        assert data["prediction"] in valid_labels

    def test_hallucination_probability_is_float(self, client):
        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        assert isinstance(data["hallucination_probability"], float)

    def test_probability_between_0_and_1(self, client):
        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        prob = data["hallucination_probability"]
        assert 0.0 <= prob <= 1.0

    def test_threshold_echoed_back(self, client):
        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)

    def test_warning_always_present(self, client):
        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        assert "warning" in data
        assert len(data["warning"]) > 0

    def test_no_hallucination_label_for_low_probability(self, client):
        """Mock returns logits for 0.1 hallucination prob → below threshold → no_annotated."""
        import torch

        output = types.SimpleNamespace(
            logits=torch.tensor([[-2.0, -10.0]])  # softmax → very low prob for class 1
        )
        client._mock_mdl.return_value = output

        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_OK},
        ).json()
        assert data["prediction"] == "no_annotated_hallucination_detected"

    def test_hallucination_label_for_high_probability(self, client):
        """Force high hallucination probability → prediction should be potential_hallucination."""
        import torch

        output = types.SimpleNamespace(
            logits=torch.tensor([[-10.0, 2.0]])  # softmax → very high prob for class 1
        )
        client._mock_mdl.return_value = output

        data = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": SAMPLE_RESPONSE_HALL},
        ).json()
        assert data["prediction"] == "potential_hallucination"
        assert data["hallucination_probability"] > 0.5


class TestVerifyValidation:
    def test_empty_evidence_returns_422(self, client):
        response = client.post(
            "/verify",
            json={"evidence": "", "response": SAMPLE_RESPONSE_OK},
        )
        assert response.status_code == 422

    def test_empty_response_returns_422(self, client):
        response = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": ""},
        )
        assert response.status_code == 422

    def test_missing_evidence_returns_422(self, client):
        response = client.post(
            "/verify",
            json={"response": SAMPLE_RESPONSE_OK},
        )
        assert response.status_code == 422

    def test_missing_response_returns_422(self, client):
        response = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE},
        )
        assert response.status_code == 422

    def test_evidence_too_long_returns_422(self, client):
        # max_length is 100_000 characters
        long_text = "a" * 100_001
        response = client.post(
            "/verify",
            json={"evidence": long_text, "response": SAMPLE_RESPONSE_OK},
        )
        assert response.status_code == 422

    def test_response_too_long_returns_422(self, client):
        # max_length is 50_000 characters
        long_text = "b" * 50_001
        response = client.post(
            "/verify",
            json={"evidence": SAMPLE_EVIDENCE, "response": long_text},
        )
        assert response.status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        response = client.post("/verify", data="not json")
        assert response.status_code in (422, 415)
