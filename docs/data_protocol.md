# Data and annotation protocol

## Audio provenance

Use only audio that is public-domain, explicitly licensed for research, or supplied by a rights holder with documented permission. Store a source identifier, license, attribution, checksum, and preprocessing record in the private data registry. Do not commit raw audio, stems, or generated audio to this repository.

The versioned manifests contain paths and task metadata only. Placeholder paths in the example manifests must be replaced locally before use.

## Benchmark construction

- Keep all stems from one original song in the same split.
- Deduplicate by source checksum before splitting.
- Reserve a test split that is never used to construct instructions, train adapters, or select checkpoints.
- Balance add, remove, and replace operations and report results separately.
- Include genre and instrumentation metadata for subgroup analysis, without using those fields as hidden evaluation labels.

## Instruction generation

Write concise instructions that identify the operation and target. For preservation analysis, explicitly name important untouched content when appropriate. Keep a held-out set of human-written instructions separate from templated instructions to test generalization.

## Preference annotation

Annotators hear the source and both candidates, read the instruction first, and choose the better overall edit. They may mark a tie. Capture criterion-level judgments for requested edit, untouched content, and audio quality, but do not expose model names or training conditions.

Report the number of annotators, agreement, tie rate, screening rules, and confidence intervals. Do not use the same listeners to tune a reward model and claim an unbiased final preference result.

## Leakage and release checklist

- Verify source checksums do not cross train/validation/test splits.
- Hash annotator IDs and remove personal information.
- Keep private audio registries separate from versioned metadata.
- Record license and consent status before external release.
- Release scripts and manifests only when every referenced asset has compatible redistribution rights.
