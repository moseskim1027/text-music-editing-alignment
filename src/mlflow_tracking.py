"""Small optional MLflow integration for experiment scripts."""

from pathlib import Path


def log_run(config: dict, metadata: dict, metrics: dict, artifact_paths: list[str] | None = None):
    """Log one run and return its MLflow run ID.

    MLflow is imported lazily so validators and metadata tools remain usable
    without the tracking dependency.
    """
    import mlflow

    with mlflow.start_run() as run:
        mlflow.log_params({key: str(value) for key, value in config.items()})
        mlflow.log_params({f"metadata.{key}": str(value) for key, value in metadata.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in artifact_paths or []:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id
