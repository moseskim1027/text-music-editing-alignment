# text-music-editing-alignment
# Text-Guided Music Editing and Post-Training Alignment

An experimental research repository for editing music with natural-language instructions and aligning generative music systems with human preferences. The project is organized to make datasets, training recipes, evaluations, and reproducible experiments easy to compare.

## Research scope

This project focuses on **preservation-aware preference alignment for text-guided music editing**. Given source music and a natural-language instruction, we will adapt a pretrained MusicGen-small-style editor with lightweight adapters so that it performs the requested stem edit while preserving non-target musical content.

The initial benchmark is intentionally narrow:

- **Add** one stem, such as “add a guitar part.”
- **Remove** one stem, such as “remove the vocals.”
- **Replace** one stem, such as “replace the piano with strings.”

The primary research question is whether preference optimization improves the balance between edit strength and preservation without requiring full-model training. The first implementation will use LoRA/adapters and short 4–8 second clips, making local experimentation possible on a 16 GB Apple-silicon machine. Larger confirmatory runs can use a cloud GPU.

The project will use Instruct-MusicGen as the principal reference direction; other systems remain literature context rather than parallel implementations.

## Runtime strategy

The training code is designed to be portable, but the execution environments differ:

- **Local M1 development and training:** run Python natively on macOS so PyTorch can access Apple MPS. Keep the BabySlakh/Slakh audio outside the repository, and use the Dockerized MLflow service for experiment tracking.
- **Cloud GPU training:** run the same code in Docker with the NVIDIA override: `make cloud-test` or the GPU Compose configuration. Cloud runs use CUDA and are the place for larger ablations.
- **Docker on macOS:** use it for tests, validation, data-manifest preparation, and MLflow. Docker’s Linux VM should not be assumed to provide MPS access to the host GPU.

Run `make device` to inspect the accelerator visible inside Docker. A future native macOS training environment will perform the corresponding MPS check outside Docker before loading MusicGen.

## Repository layout

```text
configs/     Experiment and training configurations
docs/        Design notes, protocols, and literature summaries
notebooks/   Exploratory analysis and visualization notebooks
research/    Research notes, benchmark definitions, and experiment records
scripts/     Reproducibility and evaluation entry points
src/         Reusable datasets, models, training, and evaluation code
tests/       Unit and integration tests
```

## Getting started

All commands below assume the repository root and keep audio, checkpoints, generated outputs, and MLflow state outside Git.

### 1. Build and run the Docker development environment

```bash
make build
make test
```

`make test` runs the complete suite inside the `research` Docker service. Use `make device` to inspect accelerator visibility inside Docker. Docker on macOS does not expose the host Apple MPS device; use the native environment for MPS training.

### 2. Prepare the metadata-only BabySlakh manifest

Place the locally licensed dataset outside this repository, for example:

```text
/Users/moseskim/Downloads/babyslakh_16k
```

Generate `data/derived.jsonl` without copying audio into Git:

```bash
make prepare-data \
  DATASET_ROOT=/Users/moseskim/Downloads/babyslakh_16k \
  SUBSET=data/derived.jsonl
```

The manifest records `example_id`, `source_audio`, edit instruction, operation, target stem, split, and target paths. It is metadata only; validate it with:

```bash
docker compose run --rm research python src/validate_manifest.py data/derived.jsonl
```

Do not commit `data/derived.jsonl`, audio, model weights, checkpoints, `outputs/`, or MLflow volumes.

### 3. Install and verify native macOS MPS

```bash
make native-install
make native-device
```

The native environment is required for Apple GPU access. The MusicGen checkpoint must already be available in the local Hugging Face cache because training uses `local_files_only=True`.

### 4. Run labeled MusicGen training

The compact prototype defaults to one selected example, 10 optimizer steps, MPS, and a 10-second comparison window:

```bash
MANIFEST=data/derived.jsonl \
STEPS=10 \
DEVICE=mps \
make musicgen-train
```

For a small multi-example run, leave `Example ID` empty in the UI or use the worker directly:

```bash
.venv-macos/bin/python src/train_musicgen_adapter.py data/derived.jsonl \
  --limit 3 --steps 3 --duration-seconds 10 --device mps
```

Use `--example-id Track00001_S02_add` to select the verified piano-add example. Outputs are written to ignored local paths under `outputs/runs/latest/`.

### 5. Start MLflow, API, and UI

Start MLflow (port 5000 by default):

```bash
make mlflow
```

If macOS already owns port 5000, use port 5001:

```bash
MLFLOW_PORT=5001 docker compose up -d mlflow
```

Start the native API in one terminal and the UI in another:

```bash
make api-native
make ui
```

Open `http://localhost:8080`. Use **Validate & preview run**, then **Start training**. The UI separates edit-example metadata from training settings, reports training and generation phases, and displays the original/generated audio pair and evaluation results. If MLflow uses port 5001, open `http://localhost:5001` until the UI tracking URL is configured for that port.

![Training workflow](docs/training.gif)

![Evaluation workflow](docs/evaluation.gif)

### 6. Evaluate artifacts

The backend evaluates generated artifacts automatically after generation. To run the evaluator directly:

```bash
.venv-macos/bin/python src/evaluate_audio.py \
  outputs/runs/latest/source.wav \
  outputs/runs/latest/reference_target.wav \
  outputs/runs/latest/generated_edit.wav \
  --output outputs/runs/latest/evaluation.json
```

The report contains adherence, preservation, and quality audio proxies. Preference win rate remains `N/A` until blinded human preference labels are available. Aggregate manually scored JSONL results and log them to MLflow with `make evaluate` or `python src/log_evaluation.py`.

## Evaluation plan

Every experiment should report task performance, preservation, quality, and alignment metrics on a fixed evaluation split, alongside compute and configuration details.

### Core metrics

| Dimension | Metric | Definition |
| --- | --- | --- |
| Edit success | **Edit Success Rate (ESR)** | Fraction of examples judged to satisfy the requested transformation while remaining valid audio. |
| Prompt adherence | **Text-Music Alignment (TMA)** | Similarity between the instruction and generated audio using a frozen text-audio embedding model; report mean and thresholded accuracy. |
| Edit preservation | **Preservation Score (PS)** | Similarity between source and edited audio on attributes not targeted by the instruction, measured with embeddings plus targeted attribute probes. |
| Audio quality | **Fréchet Audio Distance (FAD)** | Distance between embeddings of real reference audio and generated audio; lower is better. |
| Audio quality | **CLAP/FAD calibration set score** | Frozen-set distributional and text-audio quality checks, reported with confidence intervals. |
| Structure | **Temporal/structural consistency (TSC)** | Agreement of beat, key, tempo, segment boundaries, and duration between source and edit when those attributes are not requested to change. |
| Human alignment | **Pairwise Preference Win Rate (PPWR)** | Percentage of blind human comparisons in which a method is preferred over the baseline; report annotator agreement and bootstrap intervals. |
| Training efficiency | **Quality per compute (QpC)** | Best validated alignment/quality score divided by training FLOPs or GPU-hours, with parameter count and trainable parameter fraction. |

For each metric, the protocol will specify the evaluator version, audio normalization, segment length, aggregation rule, random seed, and statistical uncertainty. Metrics must be interpreted jointly: improving adherence at the expense of preservation or quality is not considered a complete improvement.

## Reproducibility conventions

- Pin data manifests, model checkpoints, evaluator versions, and random seeds.
- Keep source/edit/instruction triplets and train/validation/test identities auditable.
- Record hardware, wall-clock time, GPU-hours, trainable parameters, and all configuration values.
- Compare against a frozen baseline and include ablations for adapters, optimization, preservation losses, and preference objectives.

## Status

- **Repository scaffolding** — initial tree, scope, metrics, and reproducibility conventions established.
- **Preservation-aware alignment scope** — narrowed to three stem edits, adapter training, pairwise preferences, and joint edit/preservation evaluation.

The next implementation milestone is a reproducible benchmark manifest and baseline inference pipeline.

The benchmark manifest validator is available at `src/validate_manifest.py`; it checks required fields, supported operations/splits, valid JSONL, and unique example IDs. The manifest-driven baseline inference contract is available at `src/baseline_inference.py`; run it with `--dry-run` before installing the optional AudioCraft backend.

The repository is Dockerized for reproducible development and testing. Run `docker compose build` followed by `docker compose run --rm research` to execute the test suite in the pinned Python environment. The default container runs as a non-root user, and local data/checkpoints are excluded from the image.

Evaluation aggregation is available at `src/evaluate.py`. It consumes per-example JSONL scores, reports overall and per-operation means, and computes preservation-adjusted edit success.

Pairwise alignment data is validated by `src/validate_preferences.py`, which checks the preference label, candidate distinction, criteria, required metadata, and produces label-balance summaries.

The adapter-training plan and M1/16 GB starting configuration are documented in `docs/training_plan.md` and `configs/adapter_training.json`.

Example metadata-only manifests are provided in `data/benchmark.example.jsonl` and `data/preferences.example.jsonl`. Replace the placeholder audio paths with locally licensed data; raw audio remains outside version control.

Continuous integration builds the Docker image and runs the complete test suite on pushes and pull requests targeting `main`.

Experiment metadata can be recorded with `src/record_run.py`; it captures hashes for the configuration and manifest, the git commit, runtime details, and the selected device.

The audio sourcing, split, annotation, and leakage controls are specified in `docs/data_protocol.md`. Raw audio and stems are intentionally excluded from version control.

The metadata-only provenance example is in `data/registry.example.jsonl`; validate a private registry with `src/validate_registry.py` before creating benchmark manifests.

MLflow tracking is available through Docker Compose. Start it with `docker compose up -d mlflow` and wait for `docker compose ps` to report it healthy, then open `http://localhost:5000` (set `MLFLOW_PORT=5001` if that port is occupied). The SQLite database and artifacts persist in named Docker volumes. Training scripts can use `src/mlflow_tracking.py` to log configuration, run metadata, metrics, and selected artifacts to the local server.

Use the `Makefile` as the standard command wrapper: `make build`, `make test`, `make mlflow`, `make validate`, or `make evaluate`. Run `make help` to see what each target does. All Python and MLflow commands run inside Docker.

Local runs use the default container with MPS/CPU selected by the training configuration. Cloud GPU hosts can use the same image with `make cloud-test` or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm research`; the override enables NVIDIA GPU passthrough and sets `DEVICE=cuda`.

Run `make device` to report accelerator availability inside Docker before training. The probe prefers CUDA, then MPS, then CPU, and works even before PyTorch is installed.

For native M1 setup, run `make native-install` once, then `make native-device`. The installer uses the official PyPI index to obtain the arm64 wheels. The probe reports macOS GPU/Metal visibility separately from PyTorch MPS runtime availability. If macOS reports Metal but MPS is unavailable, run the command from a native terminal rather than Docker or a restricted shell. The pinned native dependencies are in `requirements-macos.txt`; the environment is ignored by Git. If an older venv already exists, rerun `make native-install` rather than installing with its old pip command.

The compact experiment dashboard is available with `make ui` at `http://localhost:8080`. It previews the shared experiment contract (manifest, edit instruction, adapter settings, device, and MLflow endpoint) without moving audio data into Git. Training execution will consume this same contract in the next UI milestone.

The dashboard control API is available with `make api` at `http://localhost:8000`. Its `/device` endpoint reports accelerator visibility and `/experiments/preview` validates a run contract before training is launched.

For actual local MusicGen training, stop the Docker API and run `make api-native`. This lets the API launch the native MPS worker with access to the local model cache and BabySlakh files; the Docker UI can continue using `http://localhost:8000`, and MLflow remains the tracking service.

The remaining implementation milestones and acceptance criteria are tracked in [`research/roadmap.md`](research/roadmap.md). The roadmap covers local-first editing, alignment, UI, tracking, data experiments, and human evaluation; cloud scaling is intentionally deferred.

Deterministic target rendering is available at `src/build_edit_targets.py`. It supports stem subtraction for remove, stem addition for add, and replacement by subtracting the target stem and adding an explicitly recorded replacement stem. Rendered audio belongs in ignored local output directories; manifests should record the source and target paths only.

The edit-aware objective layer is available at `src/edit_losses.py`. It reports edit loss, preservation loss, and their weighted total separately so experiments can quantify the requested-edit versus untouched-content trade-off.

The source-audio fusion primitive is available at `src/audio_fusion.py`. It projects pooled source-audio features into the text-conditioning dimension and prepends an audio prefix token; the model-specific encoder and MusicGen integration remain isolated in the training backend.

Run `make smoke-train` to exercise device selection and a 10-step adapter-only optimization loop. This is a runtime smoke test, not MusicGen training; it confirms the native MPS/CUDA/CPU path before the licensed MusicGen backend is connected.

Run `make musicgen-check` to verify access to `facebook/musicgen-small` and its processor without loading model weights. The checkpoint preflight deliberately precedes real training because plain MusicGen is text-to-music; the source-audio editing conditioning path must be validated before adapter optimization.

Run `make musicgen-train` from the native MPS environment for labeled MusicGen LoRA training. It uses manifest targets for supervised add/remove/replace training, supports multi-example pools and configurable 1–30 second windows, and writes ignored local artifacts under `outputs/runs/latest/`.

For the real-data phase, place a locally licensed Slakh2100 subset outside the repository and run `make prepare-data DATASET_ROOT=/path/to/slakh SUBSET=/path/to/derived.jsonl`. The Docker target mounts the dataset read-only and the output directory read/write. The tool scans mixtures and stems, preserves the official splits, and writes metadata-only edit records; it does not copy audio into the repository.

Evaluation results can be logged with `python src/log_evaluation.py path/to/scores.jsonl`; use `--tracking-uri` for a different MLflow server or `--experiment` to select an experiment. `make evaluate` uses the `text-music-editing-alignment-local` experiment by default (`MLFLOW_EXPERIMENT=...` overrides it).
