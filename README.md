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
