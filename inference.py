"""The disease models: photo in, probabilities out.

No Streamlit and no FastAPI in here, so both the Streamlit demo (app.py) and the API (api.py)
can import it.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

SHRIMP_IMG_SIZE = 224
FISH_IMG_SIZE = 300

# The order the models were trained on. Never reorder: every answer would swap.
SHRIMP_CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]
FISH_CLASSES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "EUS Disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease",
]

SHRIMP_SEVERITY = {
    "Healthy": "none",
    "BG": "moderate",
    "WSSV": "critical",
    "WSSV_BG": "critical",
}


@dataclass(frozen=True)
class Prediction:
    """One class and how likely the model thinks it is (0 to 1)."""

    label: str
    probability: float


def load_model(path: str | Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path))


def open_image(data: bytes) -> Image.Image:
    """Turn uploaded bytes into a photo, upright. Not a readable photo -> ValueError."""
    try:
        image = Image.open(io.BytesIO(data))
        # Phones store the rotation in EXIF; apply it so the model sees the shrimp upright.
        image = ImageOps.exif_transpose(image)
        # Image.open only reads the header; load() decodes the pixels, so a broken file
        # fails here and not later inside the model.
        image.load()
    except OSError as error:  # PIL's UnidentifiedImageError is an OSError
        raise ValueError("not a readable photo") from error
    return image


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Photo -> a batch of one, shape (1, size, size, 3), pixel values 0-255."""
    image = image.convert("RGB")  # PNG transparency or greyscale -> 3 colour channels
    image = image.resize((size, size))
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def classify(
    image: Image.Image, session: ort.InferenceSession, classes: list[str], size: int
) -> list[Prediction]:
    """Every class with its probability, highest first."""
    batch = preprocess(image, size)
    input_name = session.get_inputs()[0].name
    probabilities = session.run(None, {input_name: batch})[0][0]
    predictions = [
        Prediction(label, float(p)) for label, p in zip(classes, probabilities, strict=True)
    ]
    return sorted(predictions, key=lambda p: p.probability, reverse=True)


def margin(predictions: list[Prediction]) -> float:
    """How far the top answer is ahead of the second (0 = a tie, 1 = certain)."""
    if len(predictions) < 2:
        raise ValueError("margin needs at least two predictions")
    first, second = sorted((p.probability for p in predictions), reverse=True)[:2]
    return first - second