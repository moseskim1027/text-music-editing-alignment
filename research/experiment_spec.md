# Preservation-Aware Preference Alignment

## Hypothesis

Preference optimization can improve instruction adherence for stem editing, but an unconstrained objective may damage non-target content. A preservation-aware objective should improve the trade-off between target edit strength and non-target preservation.

## Controlled experiment

Compare three systems on the same source clips and instructions:

1. Frozen pretrained editor.
2. Adapter fine-tuning on edit triplets.
3. Adapter preference optimization on pairwise outputs.

Keep the base checkpoint, audio preprocessing, decoding settings, evaluation split, and random seeds fixed. Train only adapter parameters in systems 2 and 3.

## Example record

```json
{
  "example_id": "slakh_test_000001",
  "source_audio": "audio/source/slakh_test_000001.wav",
  "instruction": "Remove the vocals",
  "operation": "remove",
  "target_stem": "vocals",
  "target_audio": "audio/target/slakh_test_000001.wav",
  "untouched_stems": ["drums", "bass", "guitar"],
  "split": "test"
}
```

## Preference record

```json
{
  "example_id": "slakh_test_000001",
  "instruction": "Remove the vocals",
  "candidate_a": "audio/candidates/000001_a.wav",
  "candidate_b": "audio/candidates/000001_b.wav",
  "preferred": "b",
  "criteria": ["requested_edit", "untouched_content", "audio_quality"],
  "annotator_id": "hashed_annotator_id"
}
```

Annotators should hear the source and both candidates, read the instruction before listening, and select the better overall edit. They should also provide optional criterion-level judgments so preference errors can be analyzed rather than reduced to a single unexplained label.

## Primary metrics

- **Target Change Score (TCS):** measured change in the requested stem or attribute; higher means the edit is more evident.
- **Non-target Preservation Score (NPS):** similarity of untouched stems/features between source and output; higher is better.
- **Text-Music Alignment (TMA):** similarity between instruction and edited output from a frozen text-audio evaluator.
- **Preservation-adjusted Edit Success (PAES):** `Edit Success × NPS`, reported alongside its two components rather than as a replacement for them.
- **Pairwise Preference Win Rate (PPWR):** fraction of human comparisons won against the frozen baseline.
- **Quality:** FAD or another fixed distributional audio-quality metric, plus listening-study quality judgments.

Report bootstrap 95% confidence intervals, per-operation scores, and the TCS/NPS Pareto curve. A method is not considered an improvement if it raises TCS while causing a statistically significant drop in NPS or quality.

## Resource limits

The local development setting uses MusicGen-small, 4–8 second mono clips, batch size 1, gradient accumulation, and adapter-only updates. Full-model fine-tuning and large-scale preference collection are out of scope for local runs.
