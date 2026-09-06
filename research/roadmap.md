# Research implementation roadmap

This roadmap tracks the remaining local-first capabilities for preservation-aware text-guided music editing. Cloud CUDA scaling is intentionally out of scope for this milestone.

## Milestones

- [ ] **Edit-target construction** — derive reproducible add/remove/replace targets from BabySlakh stems, retain source/target/untouched-stem provenance, and verify no split leakage.
  - Done when: a metadata-only manifest and deterministic target builder pass train/validation/test fixture tests.
- [ ] **Edit-aware objectives** — add target-edit and non-target-preservation losses around the frozen MusicGen-small + LoRA path.
  - Done when: a one-example MPS run reports separate edit and preservation losses and both are reproducible from a fixed seed.
- [ ] **UI/API integration** — make the dashboard submit the shared experiment contract and display validation, run state, and errors.
  - Done when: a UI dry run reaches the API and returns the same validated payload as the CLI.
- [ ] **MLflow logging and artifacts** — log configuration, device, losses, metrics, checkpoints, and generated audio without placing data in Git.
  - Done when: a local run is queryable in MLflow with linked metadata and artifacts.
- [ ] **Small real-data experiment** — run one to five BabySlakh examples for 10–50 MPS steps using LoRA only.
  - Done when: the run completes within the M1 memory budget and produces a recorded baseline comparison.
- [ ] **Preference data and optimization** — generate blinded candidate pairs and implement the first preference-alignment objective.
  - Done when: preference records validate, candidate identity is hidden from annotators, and an optimization dry run consumes them.
- [ ] **Evaluation and human comparison** — produce metric reports and a small blind human comparison for adherence, preservation, quality, and preference win rate.
  - Done when: reports include evaluator versions, uncertainty, annotator agreement, and a frozen baseline comparison.

## Dependency order

Edit targets → edit-aware objectives → MLflow logging → small real-data experiment → preference data/optimization → evaluation/human comparison. UI/API integration can proceed in parallel after the experiment contract remains stable.

## Research guardrails

- Keep all raw audio, stems, checkpoints, generated audio, and MLflow state outside version control.
- Use native macOS MPS for local training; reserve Docker for services and reproducibility checks.
- Keep the base MusicGen-small model frozen and report trainable parameter counts.
- Treat reconstruction as a runtime smoke test, not evidence of successful editing or alignment.
