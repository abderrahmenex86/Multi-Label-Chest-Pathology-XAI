import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "up", "model_loaded": engine.model is not None, "target_run": engine.target_run}


@app.post("/api/v1/predict")
async def predict(request: Request):
    form = await request.form()
    upload = form["file"]
    target = form.get("target", None)

    buffer = await upload.read()

    return engine.predict(buffer, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
