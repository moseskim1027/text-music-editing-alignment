# Text-Guided Music Editing and Alignment

Research code for preservation-aware text-guided music editing. The prototype uses MusicGen-small with LoRA adapters and labeled add/remove/replace targets derived from a local Slakh/BabySlakh subset.

## Scope

The research question is whether lightweight adapter training can satisfy a requested edit while preserving non-target musical content. Raw audio, model weights, checkpoints, generated audio, manifests, and MLflow state remain outside Git.

### Current project scope

The prototype is intentionally limited to short, single-track experiments on a local Slakh/BabySlakh subset. It trains LoRA adapters on labeled source/target audio pairs for three operations: add, remove, and replace. The base MusicGen-small model remains frozen. The UI and API support local MPS runs, multi-example pools with batch size 1, generated audio artifacts, and evaluation reports.

This repository is not yet a production editor or a complete alignment system. The current audio metrics are proxies, preference win rate requires human judgments, and larger-scale batching, stronger evaluators, and preference optimization remain future work.

### Terminology

| Term | Meaning |
| --- | --- |
| Stem | An isolated musical component, such as drums, piano, bass, or vocals. |
| Mix | The combined audio waveform formed from multiple stems. |
| Source audio | The input mix supplied to the editor. |
| Target audio | The reference waveform produced by the requested edit operation. |
| Edit instruction | Natural-language text describing the requested change. |
| Add | Combine a source context with a selected stem to represent adding that part. |
| Remove | Subtract the selected stem from the mix. |
| Replace | Subtract one stem and add an explicitly supplied replacement stem. |
| Preservation | Retention of musical content not selected for editing. |
| Adapter / LoRA | A small trainable parameter update applied while the base model stays frozen. |
| Audio codes | Discrete latent tokens produced by the model’s neural audio codec and used as training targets. |
| Manifest | A JSONL metadata file that maps example IDs to audio paths, instructions, labels, and splits. |
| MPS | Apple’s Metal Performance Shaders backend for PyTorch acceleration on Apple silicon. |

## Layout

```text
configs/   Training configuration
docs/      Data protocol, training plan, and UI demonstrations
research/  Scope and roadmap
src/       Training, target construction, API, and evaluation
tests/     Dockerized test suite
ui/        Compact experiment dashboard
```

## Quick start

```bash
make build
make test
make device
```

Docker on macOS does not expose Apple MPS; use native Python for local M1/M2/M3 training.

## Prepare local data

Keep licensed BabySlakh/Slakh files outside the repository, for example `/Users/moseskim/Downloads/babyslakh_16k`. Create the metadata-only manifest with:

```bash
make prepare-data \
  DATASET_ROOT=/Users/moseskim/Downloads/babyslakh_16k \
  SUBSET=data/derived.jsonl
```

`data/derived.jsonl` records example IDs, source paths, edit instructions, operations, target stems, and splits. It contains no audio. Validate it with:

```bash
docker compose run --rm research python src/validate_manifest.py data/derived.jsonl
```

Never commit the generated manifest or audio files.

## Native MPS setup

```bash
make native-install
make native-device
```

The MusicGen-small checkpoint must already be in the local Hugging Face cache; training uses local-only model loading.

## Training

Training uses deterministic labeled edit targets. For one selected example:

```bash
.venv-macos/bin/python src/train_musicgen_adapter.py data/derived.jsonl \
  --example-id Track00001_S02_add --steps 10 --duration-seconds 10 --device mps
```

For a small multi-example pool:

```bash
.venv-macos/bin/python src/train_musicgen_adapter.py data/derived.jsonl \
  --limit 3 --steps 3 --duration-seconds 10 --device mps
```

The default UI example is `Track00001_S02_add` (S02 piano; S01 drums). Outputs are written under ignored `outputs/runs/latest/`.

## UI and API

Start the native API and Docker UI in separate terminals:

```bash
make api-native
make ui
```

Open [http://localhost:8080](http://localhost:8080). The UI validates the edit contract, reports training/generation phases, and displays original/generated audio and evaluation results.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/device
curl -s http://localhost:8000/experiments/status | python -m json.tool
```

### Training

![Training workflow](docs/training.gif)

### Evaluation

![Evaluation workflow](docs/evaluation.gif)

## MLflow

```bash
make mlflow
```

Open [http://localhost:5000](http://localhost:5000). If that port is occupied:

```bash
MLFLOW_PORT=5001 docker compose up -d mlflow
```

Then open [http://localhost:5001](http://localhost:5001). MLflow data is stored in Docker volumes.

## Evaluation

The API evaluates generated artifacts after a run. To evaluate them directly:

```bash
.venv-macos/bin/python src/evaluate_audio.py \
  outputs/runs/latest/source.wav \
  outputs/runs/latest/reference_target.wav \
  outputs/runs/latest/generated_edit.wav \
  --output outputs/runs/latest/evaluation.json
```

Reports include 0–1 adherence, preservation, and quality audio proxies; higher is better. Preference win rate remains unavailable until blinded human preference labels exist. Aggregate scored JSONL results with `make evaluate`.

| Metric | Interpretation |
| --- | --- |
| Edit Success Rate | Fraction of edits judged successful and valid |
| Text-Music Alignment | Similarity between instruction and generated audio |
| Preservation Score | Similarity of non-target content before and after editing |
| Fréchet Audio Distance | Distributional distance from reference audio; lower is better |
| Preference Win Rate | Fraction of blind comparisons preferred over the baseline |

Metrics must be reported jointly; higher adherence is not an improvement if preservation or quality decreases.

## Current capabilities

- Metadata-only labeled manifests with split-leakage validation.
- Deterministic add/remove/replace target rendering.
- MusicGen-small LoRA training on labeled targets.
- Single-example and multi-example local MPS runs.
- Dockerized API, UI, tests, and MLflow service.
- Audio artifact generation and evaluation reports.

## Future work

- Add padded mini-batching and multi-epoch training.
- Replace audio proxies with frozen text-audio and quality evaluators.
- Add explicit untouched-stem preservation loss to the training loop.
- Log each run and artifact to MLflow with a run-specific UI link.
- Collect blinded preference pairs and implement preference optimization.
- Run larger held-out experiments and human evaluation with uncertainty estimates.

See [research/roadmap.md](research/roadmap.md), [docs/data_protocol.md](docs/data_protocol.md), and [docs/training_plan.md](docs/training_plan.md) for details.
