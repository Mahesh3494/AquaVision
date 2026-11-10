"""AquaVision as a web service: POST a photo, get the disease check back as JSON.

Run it:  uv run fastapi dev api.py --port 8001   (then open http://127.0.0.1:8001/docs)
"""

import hashlib
from functools import cache
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from starlette.exceptions import HTTPException as StarletteHTTPException

import inference

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}
SHRIMP_MODEL_PATH = Path(__file__).resolve().parent / "shrimp_model.onnx"

app = FastAPI(title="AquaVision", version="1.0.0")


# ---------- the model, loaded once ----------


@cache
def shrimp_model():
    """Load the ONNX model on the first request and keep it for every request after."""
    return inference.load_model(SHRIMP_MODEL_PATH)


@cache
def shrimp_model_version() -> str:
    """File name + the start of its SHA-256: changes whenever the model file changes."""
    digest = hashlib.sha256(SHRIMP_MODEL_PATH.read_bytes()).hexdigest()[:12]
    return f"{SHRIMP_MODEL_PATH.name}@{digest}"


# ---------- response shape ----------


class CamelModel(BaseModel):
    # Python fields are snake_case; the JSON is camelCase (model_version -> modelVersion).
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ClassScore(CamelModel):
    label: str
    severity: str
    probability: float


class DetectResult(CamelModel):
    label: str
    severity: str
    confidence: float  # probability of the top answer, 0-1
    reliability: float  # margin: top probability minus the second, 0-1
    model_version: str
    top_k: list[ClassScore]


# ---------- errors: always {"error": {"code", "message"}} ----------


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
    return error_response(error.status, error.code, error.message)


@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"][1:]) or "request"
    return error_response(422, "validation_error", f"{field}: {first['msg']}")


@app.exception_handler(StarletteHTTPException)
def handle_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
    # 404 Not Found -> "not_found", 405 Method Not Allowed -> "method_not_allowed"
    code = HTTPStatus(error.status_code).phrase.lower().replace(" ", "_")
    return error_response(error.status_code, code, str(error.detail))


# ---------- endpoints ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/detect", response_model=DetectResult, response_model_by_alias=True)
def detect(
    image: Annotated[UploadFile, File()],
    species: Annotated[Literal["shrimp"], Form()],
) -> DetectResult:
    # A plain `def` (not async): FastAPI runs it in a worker thread, so a slow model run
    # doesn't block other requests.
    if image.content_type not in ALLOWED_TYPES:
        raise ApiError(415, "unsupported_media_type", "Send the photo as a JPG or PNG.")

    # Read one byte past the limit: if it's there, the file is too big. Never read it all.
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(413, "image_too_large", "The photo is over 8 MB. Send a smaller one.")

    try:
        photo = inference.open_image(data)
    except ValueError:
        raise ApiError(400, "invalid_image", "That file couldn't be read as a photo.") from None

    predictions = inference.classify(
        photo, shrimp_model(), inference.SHRIMP_CLASSES, inference.SHRIMP_IMG_SIZE
    )
    top = predictions[0]
    return DetectResult(
        label=top.label,
        severity=inference.SHRIMP_SEVERITY[top.label],
        confidence=top.probability,
        reliability=inference.margin(predictions),
        model_version=shrimp_model_version(),
        top_k=[
            ClassScore(
                label=p.label,
                severity=inference.SHRIMP_SEVERITY[p.label],
                probability=p.probability,
            )
            for p in predictions
        ],
    )
