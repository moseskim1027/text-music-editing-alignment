.PHONY: help build test cloud-test device native-install native-device ui api mlflow validate evaluate prepare-data

# Show the available repository commands.
help:
	@printf '%s\n' \
		'make build       Build the Docker images' \
		'make test        Run the complete test suite in Docker' \
		'make cloud-test  Run tests with the NVIDIA GPU Compose override' \
		'make device      Report CUDA/MPS/CPU availability' \
		'make native-install  Create the native macOS training environment' \
		'make native-device   Check MPS from native Python' \
		'make ui       Start the compact experiment dashboard on port 8080' \
		'make api      Start the experiment control API on port 8000' \
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

# Run the same tests with NVIDIA GPU passthrough for cloud hosts.
cloud-test:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm research

# Report available accelerators inside the reproducible research container.
device:
	docker compose run --rm research python src/check_device.py

# Create the native macOS virtual environment used for local MPS work.
native-install:
	python3 -m venv .venv-macos
	.venv-macos/bin/python -m pip install --upgrade pip
	.venv-macos/bin/pip install --index-url https://pypi.org/simple -r requirements-macos.txt

# Check accelerators from native macOS Python, where MPS can be visible.
native-device:
	.venv-macos/bin/python src/check_device.py

# Start MLflow in the background; override MLFLOW_PORT if 5000 is occupied.
mlflow:
	docker compose up -d --build mlflow
	docker compose ps mlflow

# Start the read-only experiment dashboard; override UI_PORT if needed.
ui:
	docker compose up -d --build ui
	docker compose ps ui

# Start the API used to validate experiment configurations and report devices.
api:
	docker compose up -d --build api
	docker compose ps api

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
	@dataset_root="$${DATASET_ROOT:?set DATASET_ROOT}"; \
	output="$${SUBSET:-data/derived.jsonl}"; \
	output_dir="$$(dirname "$$output")"; \
	docker compose run --rm \
		-v "$$dataset_root:$$dataset_root:ro" \
		-v "$$output_dir:$$output_dir" \
		research python src/prepare_slakh_manifest.py "$$dataset_root" "$$output"
