"""FastAPI service for multi-label chest pathology inference."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from . import engine

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def cors_origins() -> list[str]:
    configured = os.getenv(
        "CHEST_XAI_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["http://localhost:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine.configure(
        run=os.getenv("CHEST_XAI_RUN") or None,
        artifacts=os.getenv("CHEST_XAI_ARTIFACTS", "artifacts"),
    )
    yield


app = FastAPI(
    title="Multi-Label Chest Pathology XAI",
    description="Research prototype for chest X-ray classification and HiResCAM explanations.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return engine.health_payload()


@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    target: str | None = Form(None),
) -> dict:
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported.",
        )

    max_bytes = int(os.getenv("CHEST_XAI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES))
    buffer = await file.read(max_bytes + 1)
    if len(buffer) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
        )
    if not buffer:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        return await run_in_threadpool(engine.predict, buffer, target)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run", default=None, help="Run directory to serve.")
    parser.add_argument("--artifacts", default="artifacts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run:
        os.environ["CHEST_XAI_RUN"] = args.run
    os.environ["CHEST_XAI_ARTIFACTS"] = args.artifacts
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
