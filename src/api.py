"""Small control-plane API for the experiment dashboard."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from threading import Thread
from pathlib import Path
import json
import subprocess
import sys
from pydantic import BaseModel, Field

from src.check_device import probe

app = FastAPI(title="Music Edit Lab API", version="0.1.0")
app.mount("/artifacts", StaticFiles(directory="outputs", check_dir=False), name="artifacts")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080"], allow_methods=["GET", "POST"], allow_headers=["*"])
run_state = {"status": "idle", "step": 0, "steps": 0, "result": None}


class ExperimentRequest(BaseModel):
    manifest: str
    example_id: str | None = None
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
        run_state.update(status="running", phase="training", step=0, steps=request.steps, result=None)
        process = subprocess.Popen(
            [sys.executable, "src/train_musicgen_adapter.py", request.manifest, "--example-id", request.example_id or "", "--instruction", request.instruction, "--steps", str(request.steps), "--device", request.device],
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
                elif event.get("event") == "phase":
                    run_state.update(phase=event["phase"])
            except (ValueError, TypeError, KeyError):
                continue
        error = process.stderr.read() if process.stderr else ""
        process.wait()
        output = "".join(output_lines)
        if process.returncode:
            raise RuntimeError(error[-2000:] or "training worker failed")
        run_state.update(phase="evaluating")
        artifact_dir = Path("outputs/runs/latest")
        evaluation = subprocess.run(
            [sys.executable, "src/evaluate_audio.py", str(artifact_dir / "source.wav"), str(artifact_dir / "reference_target.wav"), str(artifact_dir / "generated_edit.wav"), "--output", str(artifact_dir / "evaluation.json")],
            capture_output=True, text=True, check=True,
        )
        run_state["result"] = {"output": output, "evaluation": json.loads(evaluation.stdout)}
        run_state.update(status="completed", phase="completed", step=request.steps)
    except Exception as exc:
        run_state.update(status="failed", phase="failed", result={"error": str(exc)})


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
