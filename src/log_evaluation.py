#!/usr/bin/env python3
"""Aggregate evaluation scores and log them to MLflow."""

import argparse
import json
from pathlib import Path

try:
    from src.evaluate import aggregate, read_scores
except ModuleNotFoundError:  # direct execution: python src/log_evaluation.py ...
    from evaluate import aggregate, read_scores


def log_evaluation(scores_path: Path, experiment: str, tracking_uri: str) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    summary = aggregate(read_scores(scores_path))
    with mlflow.start_run() as run:
        means = summary["mean"]
        mlflow.log_metrics({f"mean_{key}": value for key, value in means.items()})
        mlflow.log_metric("preservation_adjusted_edit_success", summary["preservation_adjusted_edit_success"])
        mlflow.log_param("score_records", summary["records"])
        mlflow.log_artifact(str(scores_path), artifact_path="evaluation")
        return run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--experiment", default="text-music-editing-alignment")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    args = parser.parse_args()
    print(json.dumps({"run_id": log_evaluation(args.scores, args.experiment, args.tracking_uri)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
