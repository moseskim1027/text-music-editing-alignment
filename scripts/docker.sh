#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  echo "Usage: $0 {build|test|mlflow|validate|evaluate} [args...]" >&2
  exit 2
}

command="${1:-}"
shift || true

case "$command" in
  build)
    docker compose build
    ;;
  test)
    docker compose run --rm research
    ;;
  mlflow)
    docker compose up -d mlflow
    docker compose ps mlflow
    ;;
  validate)
    docker compose run --rm research python src/validate_manifest.py "${1:?manifest path required}"
    ;;
  evaluate)
    docker compose run --rm research python src/log_evaluation.py "${1:?scores path required}" \
      --tracking-uri "${MLFLOW_TRACKING_URI:-http://mlflow:5000}" "${@:2}"
    ;;
  *)
    usage
    ;;
esac
