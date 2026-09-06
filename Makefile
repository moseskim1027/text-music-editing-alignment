.PHONY: help build test mlflow validate evaluate prepare-data

# Show the available repository commands.
help:
	@printf '%s\n' \
		'make build       Build the Docker images' \
		'make test        Run the complete test suite in Docker' \
		'make mlflow      Start the local MLflow tracking server' \
		'make validate    Validate the example benchmark manifest' \
		'make evaluate    Log example evaluation scores to MLflow' \
		'make prepare-data DATASET_ROOT=/path/to/slakh SUBSET=data/derived.jsonl  Create edit metadata'

# Build the research and MLflow Docker images.
build:
	docker compose build

# Run all unit and integration tests in the research container.
test:
	docker compose run --rm research

# Start MLflow in the background; override MLFLOW_PORT if 5000 is occupied.
mlflow:
	docker compose up -d --build mlflow
	docker compose ps mlflow

# Validate the checked-in metadata-only benchmark fixture.
validate:
	docker compose run --rm research python src/validate_manifest.py data/benchmark.example.jsonl

# Log the checked-in evaluation fixture to the Dockerized MLflow service.
evaluate:
	docker compose run --rm research python src/log_evaluation.py data/scores.example.jsonl \
		--tracking-uri "$${MLFLOW_TRACKING_URI:-http://mlflow:5000}" \
		--experiment "$${MLFLOW_EXPERIMENT:-text-music-editing-alignment-local}"

# Scan a local Slakh-style subset and create metadata-only edit records.
prepare-data:
	docker compose run --rm research python src/prepare_slakh_manifest.py "$${DATASET_ROOT:?set DATASET_ROOT}" "$${SUBSET:-data/derived.jsonl}"
