"""Small control-plane API for the experiment dashboard."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.check_device import probe

app = FastAPI(title="Music Edit Lab API", version="0.1.0")


class ExperimentRequest(BaseModel):
    manifest: str
    operation: str = Field(pattern="^(add|remove|replace)$")
    instruction: str = Field(min_length=1, max_length=2000)
    adapter_rank: int = Field(default=8, ge=1, le=64)
    steps: int = Field(default=10, ge=1, le=100000)
    device: str = Field(default="cpu", pattern="^(cpu|mps|cuda)$")
    tracking_uri: str = "http://localhost:5000"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/device")
def device() -> dict:
    return probe()


@app.post("/experiments/preview")
def preview(request: ExperimentRequest) -> dict:
    available = probe()
    if request.device == "mps" and not available["mps"]:
        raise HTTPException(status_code=409, detail="MPS is not available in this process")
    if request.device == "cuda" and not available["cuda"]:
        raise HTTPException(status_code=409, detail="CUDA is not available in this process")
    return {"valid": True, "experiment": request.model_dump()}
