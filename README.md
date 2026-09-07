# Text-Guided Music Editing and Alignment

Research code for preservation-aware text-guided music editing. The prototype uses MusicGen-small with LoRA adapters and labeled add/remove/replace targets derived from a local Slakh/BabySlakh subset.

The goal is to make a requested musical change while retaining the parts of the source that the instruction does not target. The repository covers the experimental path from manifest creation and deterministic target construction through adapter training, local inference, evaluation, and run tracking.

## Scope

The research question is whether lightweight adapter training can satisfy a requested edit while preserving non-target musical content. Raw audio, model weights, checkpoints, generated audio, manifests, and MLflow state remain outside Git.

This is a research prototype rather than a production audio editor. It currently focuses on short, mono, stem-based examples and three edit operations:

- **Add:** introduce a target stem into the source mixture.
- **Remove:** subtract a target stem while preserving the remaining mixture.
- **Replace:** exchange one target stem for another.

Target waveforms are built deterministically from source stems. MusicGen encodes the edit instruction, the pretrained base model remains frozen, and LoRA adapters provide the trainable parameters. Current adherence, preservation, and quality scores are useful experiment proxies, not comprehensive perceptual or semantic metrics.

## How it fits together

```text
Slakh/BabySlakh stems
        |
        v
metadata-only manifest --> deterministic source/target pairs
        |                              |
        +------------------------------+
                       |
                       v
        MusicGen-small + LoRA training
                       |
                       v
       generated audio + evaluation report
                       |
                       v
              UI / API / MLflow
```

The manifest is the contract between data preparation, training, and evaluation. It identifies the source audio, edit instruction, operation, target stem, expected target audio, and data split. Keeping this metadata separate from licensed audio makes the code and example schemas shareable without redistributing the dataset.

## Model architecture

`facebook/musicgen-small` is a composite conditional-generation model with three pretrained components:

```text
edit instruction ──> frozen T5 encoder ───────────────┐
                                                      v
source waveform ──> frozen 32 kHz EnCodec ──> audio codes
                                                      |
                                                      v
                                      MusicGen Transformer decoder
                                      24 layers, width 1024, 16 heads
                                      4 delayed/interleaved codebooks
                                                      |
                                                      v
                                        predicted audio codes
                                                      |
                                                      v
                                      frozen EnCodec decoder ──> waveform
```

The T5 encoder converts the instruction into text hidden states. EnCodec converts a mono 32 kHz waveform into four streams of discrete codes sampled at 50 frames per second. The 300M-parameter MusicGen decoder autoregressively predicts the four delayed code streams, attending both to prior audio codes and to the text states. EnCodec then decodes the predicted codes into audio.

In this repository, the source mixture is supplied as an audio prompt and the deterministic edited waveform supplies the supervised target codes. The current generation path is therefore source-conditioned continuation, not a native arbitrary-span editor: it retains the generated continuation and compares it with a same-duration reference edit. That distinction is important when interpreting preservation scores and motivates the architecture directions below.

### Where LoRA is applied

The implementation in `src/train_musicgen_adapter.py` wraps the complete model with PEFT but targets modules named only `q_proj` and `v_proj`. In the pinned Transformers implementation, those names occur in the MusicGen decoder—not in T5 or EnCodec. An adapter is inserted into four projections per decoder layer:

```text
MusicGen decoder layer (repeated 24 times)
├── causal self-attention
│   ├── q_proj + LoRA rank 8
│   └── v_proj + LoRA rank 8
└── text cross-attention
    ├── q_proj + LoRA rank 8
    └── v_proj + LoRA rank 8
```

This gives 96 adapted linear projections and 1,572,864 trainable parameters with `r=8`, `lora_alpha=16`, and dropout `0.05`. The pretrained projection weights, T5 text encoder, EnCodec audio codec, embeddings, feed-forward blocks, attention key/output projections, and output heads remain frozen. Self-attention adapters can change how the model uses the source and previously generated audio tokens; cross-attention adapters can change how strongly and where the instruction affects generation.

## Layout

```text
configs/   Training configuration
docs/      Data protocol, training plan, and UI demonstrations
research/  Scope and roadmap
src/       Training, target construction, API, and evaluation
tests/     Dockerized test suite
ui/        Compact experiment dashboard
```

Notable entry points include `src/prepare_slakh_manifest.py` for dataset indexing, `src/train_musicgen_adapter.py` for labeled LoRA training, `src/evaluate_audio.py` for artifact-level evaluation, and `src/api.py` for experiment control.

## Prerequisites

- Docker with Compose for the reproducible test, API, UI, and MLflow services.
- Python 3 and Apple Silicon for the native MPS workflow, or a CUDA-capable environment with the GPU Compose override.
- A local, appropriately licensed Slakh/BabySlakh-style dataset.
- A locally cached `facebook/musicgen-small` checkpoint for offline training.

## Quick start

```bash
make build
make test
make device
```

Docker on macOS does not expose Apple MPS; use native Python for local M1/M2/M3 training.

Useful commands are discoverable with `make help`. Run the GPU-backed test container on a compatible cloud host with `make cloud-test`.

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

The checked-in files under `data/*.example.jsonl` document the expected schemas and support validation and evaluation tests without requiring the private dataset. See [the data protocol](docs/data_protocol.md) for provenance, split, licensing, and preference-annotation rules.

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

A run produces source and reference audio, the generated edit, and an evaluation JSON report. These artifacts are intentionally ignored because they may be large or derived from licensed source material.

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

For reproducible comparisons, keep the base checkpoint, preprocessing, decoding settings, data split, and random seed fixed. Report results by edit operation as well as in aggregate, and use held-out songs rather than stems from songs seen during training.

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

### Architecture directions

The current LoRA baseline is intentionally small and useful for measuring whether decoder attention alone can learn the edit task. Stronger editing architectures should be compared against it rather than assumed to be improvements:

- **Source-aligned latent editor:** encode the source and target into time-aligned EnCodec codes, then condition each target position directly on the corresponding source codes. A learned copy/edit gate could preserve unchanged codes and regenerate only regions affected by the instruction.
- **Masked infilling instead of continuation:** train a bidirectional or span-denoising latent model to replace selected time/codebook regions. This better matches edits inside an existing clip than causal continuation does.
- **Residual edit prediction:** predict a sparse latent delta or edit mask relative to the source representation, with an identity objective outside the target stem or time region. This makes preservation part of the architecture rather than only an evaluation metric.
- **Stem-aware conditioning:** retain separated or grouped source-stem embeddings during training and provide the target instrument and operation as structured conditioning. At inference, a separator could supply those streams when stems are unavailable.
- **Broader, measured adapter placement:** ablate LoRA on key/output attention projections, feed-forward layers, and source-conditioning projections; vary rank by layer; and compare against the current query/value-only budget at a fixed trainable-parameter count.

Each direction needs operation-stratified ablations and listening tests. In particular, an architecture should count as better only if edit adherence improves without degrading untouched-content preservation, audio quality, or inference cost beyond the stated budget.

See [research/roadmap.md](research/roadmap.md), [docs/data_protocol.md](docs/data_protocol.md), and [docs/training_plan.md](docs/training_plan.md) for details.

## References

- Copet, J. et al. (2023). [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284). The MusicGen architecture, codebook-delay pattern, conditioning, and model scaling.
- Défossez, A. et al. (2022). [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438). The EnCodec neural audio codec used to represent waveforms as discrete tokens.
- Hu, E. J. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). The parameter-efficient adaptation method used by the training path.
- Manilow, E. et al. (2019). [Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity](https://www.merl.com/publications/TR2019-124). The source dataset and aligned multitrack construction.
- Hugging Face. [MusicGen model documentation](https://huggingface.co/docs/transformers/model_doc/musicgen) and [`facebook/musicgen-small` model card](https://huggingface.co/facebook/musicgen-small). The composite Transformers implementation and checkpoint details used here.
