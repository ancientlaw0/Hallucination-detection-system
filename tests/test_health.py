"""
test_health.py — Tests for the basic health and information endpoints.

Endpoints covered:
  GET /          → API is alive
  GET /health    → model is loaded, device is reported
  GET /model-info → threshold, model name, task description
"""


class TestRoot:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_has_message_key(self, client):
        data = client.get("/").json()
        assert "message" in data

    def test_message_is_string(self, client):
        data = client.get("/").json()
        assert isinstance(data["message"], str)


class TestHealth:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_model_loaded_is_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_device_field_present(self, client):
        data = client.get("/health").json()
        assert "device" in data
        assert isinstance(data["device"], str)


class TestModelInfo:
    def test_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_has_threshold(self, client):
        data = client.get("/model-info").json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)

    def test_threshold_between_0_and_1(self, client):
        data = client.get("/model-info").json()
        assert 0.0 <= data["threshold"] <= 1.0

    def test_has_model_field(self, client):
        data = client.get("/model-info").json()
        assert "model" in data

    def test_has_task_field(self, client):
        data = client.get("/model-info").json()
        assert "task" in data


class TestUI:
    def test_ui_returns_200(self, client):
        response = client.get("/ui")
        assert response.status_code == 200

    def test_ui_returns_html(self, client):
        response = client.get("/ui")
        assert "text/html" in response.headers.get("content-type", "")

    def test_ui_contains_title(self, client):
        response = client.get("/ui")
        assert "Hallucination Detection" in response.text

