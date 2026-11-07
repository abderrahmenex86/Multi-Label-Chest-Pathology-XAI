from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import engine, main


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(engine, "configure", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "predictor", None)
    monkeypatch.setattr(engine, "target_run", None)
    monkeypatch.setattr(engine, "architecture", None)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_degraded_without_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_predict_rejects_non_image_content_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("example.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400


def test_predict_returns_payload_for_valid_image(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        engine,
        "predict",
        lambda buffer, target: {
            "predictions": [
                {"name": "Effusion", "probability": 0.8, "threshold": 0.5, "positive": True}
            ],
            "heatmap": "aGVhdG1hcA==",
            "target": "Effusion",
            "run": "test-run",
            "architecture": "densenet121",
        },
    )

    response = client.post(
        "/api/v1/predict",
        files={"file": ("scan.png", png_bytes(), "image/png")},
        data={"target": "Effusion"},
    )

    assert response.status_code == 200
    assert response.json()["target"] == "Effusion"
    assert response.json()["predictions"][0]["positive"] is True


def test_predict_returns_503_when_model_is_missing(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("scan.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 503


def test_predict_enforces_upload_limit(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("CHEST_XAI_MAX_UPLOAD_BYTES", "10")

    response = client.post(
        "/api/v1/predict",
        files={"file": ("scan.png", b"x" * 100, "image/png")},
    )

    assert response.status_code == 413
