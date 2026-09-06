# text-music-editing-alignment
# Text-Guided Music Editing and Post-Training Alignment

An experimental research repository for editing music with natural-language instructions and aligning generative music systems with human preferences. The project is organized to make datasets, training recipes, evaluations, and reproducible experiments easy to compare.

## Research scope

Initial directions:

- **Instruct-MusicGen** — instruction-conditioned music editing and generation.
- **SAO-Instruct** — instruction following for sound and music transformation.
- **Stable Audio Control** — controllable generation and edit-oriented conditioning.
- **InstructME** — multimodal/instruction-guided music editing.
- **MMEdit** — multimodal editing, preservation, and controllability.

We will study improvements in optimization methods, adapter and parameter-efficient training, edit preservation, prompt adherence, perceptual sound quality, and human-preference alignment.

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

The initial repository tree and research evaluation framework are established. Subsequent capabilities will be documented here as they become reproducible.
