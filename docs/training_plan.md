# Adapter training plan

The first trainable experiment is deliberately small enough to develop on an M1 Mac with 16 GB unified memory. It adapts a pretrained MusicGen-small editor rather than training a generator from scratch.

## Three-stage comparison

1. **Frozen baseline:** run the pretrained editor with fixed decoding settings.
2. **Edit adapter:** train LoRA parameters on source/instruction/target triplets.
3. **Preference adapter:** continue from stage 2 using validated pairwise preferences, with preservation included in candidate selection and analysis.

The configuration in `configs/adapter_training.json` is a starting point, not a claim that every AudioCraft operation is supported by MPS. Record peak memory, wall-clock time, skipped/fallback operators, and effective batch size for every run.

## Local constraints

- Use 4–8 second mono clips and batch size 1.
- Accumulate gradients instead of increasing the batch size.
- Keep the base model frozen and checkpoint only adapter weights.
- Start with 100–500 examples for pipeline validation before scaling.
- Use CPU fallback only for unsupported operations; mark such runs clearly.
- Run final comparisons with identical seeds, decoding parameters, and evaluation examples.

## Required run metadata

Each run should save the configuration, git commit, model identifier, dataset manifest hash, seed, device, peak memory, duration, trainable parameter count, and evaluation JSONL.
