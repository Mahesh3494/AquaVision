"""Tests for api.py: POST /v1/detect and GET /health."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import MAX_IMAGE_BYTES, app

SAMPLES = Path(__file__).resolve().parent.parent / "static" / "samples"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def sample(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


def detect(client, data: bytes, content_type="image/jpeg", species="shrimp"):
    form = {} if species is None else {"species": species}
    return client.post("/v1/detect", files={"image": ("photo.jpg", data, content_type)}, data=form)


def assert_error(response, status: int, code: str):
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


# ---------- health ----------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------- a good photo ----------


def test_detect_returns_the_result_shape(client):
    response = detect(client, sample("shrimp_bg.jpg"))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "severity", "confidence", "reliability", "modelVersion", "topK"}


def test_detect_gets_the_sample_right_with_its_severity(client):
    body = detect(client, sample("shrimp_bg.jpg")).json()
    assert body["label"] == "BG"
    assert body["severity"] == "moderate"


def test_detect_critical_sample(client):
    body = detect(client, sample("shrimp_wssv.jpg")).json()
    assert body["label"] == "WSSV"
    assert body["severity"] == "critical"


def test_top_k_has_every_class_highest_first(client):
    top_k = detect(client, sample("shrimp_bg.jpg")).json()["topK"]
    assert sorted(item["label"] for item in top_k) == ["BG", "Healthy", "WSSV", "WSSV_BG"]
    probabilities = [item["probability"] for item in top_k]
    assert probabilities == sorted(probabilities, reverse=True)
    assert set(top_k[0]) == {"label", "severity", "probability"}


def test_confidence_is_the_top_probability_and_reliability_is_the_margin(client):
    body = detect(client, sample("shrimp_bg.jpg")).json()
    first, second = body["topK"][0]["probability"], body["topK"][1]["probability"]
    assert body["confidence"] == pytest.approx(first)
    assert body["reliability"] == pytest.approx(first - second)
    assert 0 <= body["reliability"] <= body["confidence"] <= 1


def test_model_version_names_the_model_file_and_its_hash(client):
    version = detect(client, sample("shrimp_bg.jpg")).json()["modelVersion"]
    name, _, digest = version.partition("@")
    assert name == "shrimp_model.onnx"
    assert len(digest) == 12


def test_detect_accepts_png(client):
    buf = io.BytesIO()
    Image.open(SAMPLES / "shrimp_healthy.jpg").save(buf, format="PNG")
    response = detect(client, buf.getvalue(), content_type="image/png")
    assert response.status_code == 200
    assert response.json()["label"] == "Healthy"


# ---------- what the form must contain ----------


def test_species_other_than_shrimp_is_refused(client):
    # Only the shrimp model is served for now; AquaManage v1 is shrimp only.
    assert_error(detect(client, sample("shrimp_bg.jpg"), species="fish"), 422, "validation_error")


def test_missing_species_is_refused(client):
    assert_error(detect(client, sample("shrimp_bg.jpg"), species=None), 422, "validation_error")


def test_missing_image_is_refused(client):
    response = client.post("/v1/detect", data={"species": "shrimp"})
    assert_error(response, 422, "validation_error")


# ---------- bad files ----------


def test_a_file_that_is_not_jpeg_or_png_is_415(client):
    assert_error(detect(client, b"hello", content_type="text/plain"), 415, "unsupported_media_type")


def test_a_gif_is_415(client):
    assert_error(detect(client, b"GIF89a", content_type="image/gif"), 415, "unsupported_media_type")


def test_bytes_that_only_claim_to_be_a_jpeg_are_400(client):
    assert_error(detect(client, b"not really a photo"), 400, "invalid_image")


def test_an_empty_file_is_400(client):
    assert_error(detect(client, b""), 400, "invalid_image")


def test_a_photo_over_8_mb_is_413(client):
    assert MAX_IMAGE_BYTES == 8 * 1024 * 1024
    assert_error(detect(client, b"\xff" * (MAX_IMAGE_BYTES + 1)), 413, "image_too_large")


def test_exactly_8_mb_is_not_refused_for_size(client):
    # At the limit the size check passes; these bytes then fail as a photo instead.
    assert_error(detect(client, b"\xff" * MAX_IMAGE_BYTES), 400, "invalid_image")


# ---------- every error has the same shape ----------


def test_unknown_path_is_404_with_the_error_shape(client):
    assert_error(client.get("/v1/nothing-here"), 404, "not_found")


def test_wrong_method_is_405_with_the_error_shape(client):
    assert_error(client.get("/v1/detect"), 405, "method_not_allowed")
