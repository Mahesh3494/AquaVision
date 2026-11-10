"""Tests for inference.py: the model code, with no Streamlit and no FastAPI.

The Streamlit app and the FastAPI service will both import inference.py, so this is the one
place that knows class order, image size and how a photo becomes model input.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import inference
from inference import Prediction

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "static" / "samples"


@pytest.fixture(scope="session")
def shrimp_model():
    # Loading an ONNX model takes a moment: load it once for the whole test run.
    return inference.load_model(ROOT / "shrimp_model.onnx")


def jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# ---------- class lists and severity ----------


def test_shrimp_classes_are_in_the_order_the_model_was_trained_on():
    # Changing this order silently swaps every answer: WSSV would be reported as Healthy.
    assert inference.SHRIMP_CLASSES == ["BG", "Healthy", "WSSV", "WSSV_BG"]


def test_every_shrimp_class_has_a_severity():
    assert inference.SHRIMP_SEVERITY == {
        "Healthy": "none",
        "BG": "moderate",
        "WSSV": "critical",
        "WSSV_BG": "critical",
    }


def test_fish_classes_still_exist_for_the_streamlit_demo():
    assert len(inference.FISH_CLASSES) == 8
    assert "Healthy Fish" in inference.FISH_CLASSES


def test_image_sizes():
    assert inference.SHRIMP_IMG_SIZE == 224
    assert inference.FISH_IMG_SIZE == 300


# ---------- open_image: bytes from an upload -> a PIL image ----------


def test_open_image_reads_jpeg_bytes():
    image = inference.open_image((SAMPLES / "shrimp_bg.jpg").read_bytes())
    assert isinstance(image, Image.Image)


def test_open_image_reads_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(buf, format="PNG")
    assert inference.open_image(buf.getvalue()).size == (10, 10)


def test_open_image_rejects_bytes_that_are_not_a_photo():
    with pytest.raises(ValueError):
        inference.open_image(b"this is not a photo")


def test_open_image_rejects_empty_bytes():
    with pytest.raises(ValueError):
        inference.open_image(b"")


def test_open_image_applies_the_phone_rotation():
    # Phones save a portrait photo as landscape pixels plus an EXIF "rotate me" tag.
    # 0x0112 is the EXIF Orientation tag; 6 means "turn 90 degrees to show it upright".
    image = Image.new("RGB", (40, 20), "white")
    exif = image.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    assert inference.open_image(buf.getvalue()).size == (20, 40)


# ---------- preprocess: PIL image -> model input ----------


def test_preprocess_gives_one_image_batch_of_the_right_shape():
    array = inference.preprocess(Image.new("RGB", (640, 480)), 224)
    assert array.shape == (1, 224, 224, 3)
    assert array.dtype == np.float32


def test_preprocess_keeps_pixel_values_0_to_255():
    # The Keras EfficientNet model rescales inside itself: don't divide by 255 here.
    array = inference.preprocess(Image.new("RGB", (8, 8), (255, 0, 0)), 224)
    assert array.max() == 255.0
    assert array[0, 0, 0].tolist() == [255.0, 0.0, 0.0]


def test_preprocess_turns_png_transparency_and_greyscale_into_rgb():
    assert inference.preprocess(Image.new("RGBA", (8, 8)), 224).shape == (1, 224, 224, 3)
    assert inference.preprocess(Image.new("L", (8, 8)), 224).shape == (1, 224, 224, 3)


# ---------- classify: the model's answer for every class ----------


def test_classify_returns_every_class_highest_first(shrimp_model):
    image = Image.open(SAMPLES / "shrimp_bg.jpg")
    predictions = inference.classify(
        image, shrimp_model, inference.SHRIMP_CLASSES, inference.SHRIMP_IMG_SIZE
    )
    assert sorted(p.label for p in predictions) == sorted(inference.SHRIMP_CLASSES)
    probabilities = [p.probability for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_classify_probabilities_are_0_to_1_and_add_up_to_1(shrimp_model):
    image = Image.open(SAMPLES / "shrimp_healthy.jpg")
    predictions = inference.classify(
        image, shrimp_model, inference.SHRIMP_CLASSES, inference.SHRIMP_IMG_SIZE
    )
    assert all(0.0 <= p.probability <= 1.0 for p in predictions)
    assert sum(p.probability for p in predictions) == pytest.approx(1.0, abs=1e-3)
    assert all(isinstance(p.probability, float) for p in predictions)  # plain float, not numpy


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("shrimp_healthy.jpg", "Healthy"),
        ("shrimp_bg.jpg", "BG"),
        ("shrimp_wssv.jpg", "WSSV"),
        ("shrimp_wssv_bg.jpg", "WSSV_BG"),
    ],
)
def test_classify_gets_the_sample_photos_right(shrimp_model, sample, expected):
    # The samples come from the training data, so this proves the wiring (class order,
    # size, preprocessing), not that the model is good on farm photos.
    image = Image.open(SAMPLES / sample)
    predictions = inference.classify(
        image, shrimp_model, inference.SHRIMP_CLASSES, inference.SHRIMP_IMG_SIZE
    )
    assert predictions[0].label == expected


def test_classify_works_on_an_uploaded_photo_end_to_end(shrimp_model):
    data = jpeg_bytes(Image.open(SAMPLES / "shrimp_wssv.jpg"))
    image = inference.open_image(data)
    predictions = inference.classify(
        image, shrimp_model, inference.SHRIMP_CLASSES, inference.SHRIMP_IMG_SIZE
    )
    assert predictions[0].label == "WSSV"


# ---------- margin: how far the top answer is ahead of the second ----------


def test_margin_is_top_probability_minus_second():
    predictions = [Prediction("BG", 0.7), Prediction("WSSV", 0.2), Prediction("Healthy", 0.1)]
    assert inference.margin(predictions) == pytest.approx(0.5)


def test_margin_does_not_depend_on_the_order_given():
    predictions = [Prediction("Healthy", 0.1), Prediction("WSSV", 0.2), Prediction("BG", 0.7)]
    assert inference.margin(predictions) == pytest.approx(0.5)


def test_margin_is_0_for_a_tie():
    predictions = [Prediction("BG", 0.45), Prediction("WSSV", 0.45), Prediction("Healthy", 0.1)]
    assert inference.margin(predictions) == pytest.approx(0.0)


def test_margin_needs_at_least_two_predictions():
    with pytest.raises(ValueError):
        inference.margin([Prediction("BG", 1.0)])


def test_prediction_is_frozen():
    prediction = Prediction("BG", 0.5)
    with pytest.raises(AttributeError):
        prediction.label = "WSSV"  # type: ignore[misc]
