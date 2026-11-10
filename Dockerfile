# Recipe for the AquaVision API image (api.py). The Streamlit demo is not in it.
# Same layout as AquaManage's backend image: slow steps first, so Docker can reuse them.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app

# 1) Dependencies: only the API's own list. --no-default-groups skips dev (pytest, ruff)
#    and ui (Streamlit), which would add hundreds of MB the API never uses.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups

# 2) The model (17 MB, rarely changes), then the code (changes most).
COPY shrimp_model.onnx ./
COPY inference.py api.py ./

ENV PATH="/app/.venv/bin:$PATH"

# Run as an ordinary user, not root: if someone ever broke in through the API,
# they couldn't change anything in the image.
RUN useradd --create-home aquavision
USER aquavision

EXPOSE 8000

CMD ["fastapi", "run", "api.py", "--host", "0.0.0.0", "--port", "8000"]
