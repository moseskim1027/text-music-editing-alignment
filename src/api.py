"""Small control-plane API for the experiment dashboard."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import json
import subprocess
import sys
from pydantic import BaseModel, Field

from src.check_device import probe

app = FastAPI(title="Music Edit Lab API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080"], allow_methods=["GET", "POST"], allow_headers=["*"])
run_state = {"status": "idle", "step": 0, "steps": 0, "result": None}


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


def _run_training(request: ExperimentRequest) -> None:
    try:
        run_state.update(status="running", step=0, steps=request.steps, result=None)
        process = subprocess.Popen(
            [sys.executable, "src/train_musicgen_adapter.py", request.manifest, "--steps", str(request.steps), "--device", request.device],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output_lines = []
        for line in process.stdout or ():
            output_lines.append(line)
            try:
                event = json.loads(line)
                if event.get("event") == "progress":
                    run_state.update(step=event["step"], steps=event["steps"], loss=event["loss"])
            except (ValueError, TypeError, KeyError):
                continue
        error = process.stderr.read() if process.stderr else ""
        process.wait()
        output = "".join(output_lines)
        if process.returncode:
            raise RuntimeError(error[-2000:] or "training worker failed")
        run_state["result"] = {"output": output}
        run_state.update(status="completed", step=request.steps)
    except Exception as exc:
        run_state.update(status="failed", result={"error": str(exc)})


@app.post("/experiments/start")
def start(request: ExperimentRequest) -> dict:
    if run_state["status"] == "running":
        raise HTTPException(status_code=409, detail="An experiment is already running")
    if not __import__("pathlib").Path(request.manifest).exists():
        raise HTTPException(status_code=400, detail=f"Manifest not found: {request.manifest}")
    Thread(target=_run_training, args=(request,), daemon=True).start()
    return {"accepted": True, "status": "running"}


@app.get("/experiments/status")
def experiment_status() -> dict:
    return run_state
