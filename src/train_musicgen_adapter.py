#!/usr/bin/env python3
"""One-example source-conditioned MusicGen LoRA edit training run."""

import argparse
import json
from pathlib import Path

import torch
from transformers import logging

logging.set_verbosity_error()


def normalize_decoder_start_token(model):
    """Bridge the nested MusicGen config used by Transformers 4.49."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = base.config
    if getattr(config.decoder, "decoder_start_token_id", None) is None:
        config.decoder.decoder_start_token_id = config.decoder.bos_token_id
    # Keep both locations populated for generation and supervised forward paths.
    if getattr(config, "decoder_start_token_id", None) is None:
        config.decoder_start_token_id = config.decoder.decoder_start_token_id
    return model


def load_audio(path: Path, target_samples: int, sf, torchaudio) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wave = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != 32000:
        wave = torchaudio.functional.resample(wave, sample_rate, 32000)
    return wave[:, :target_samples]


def resolve_target_stem(record: dict) -> Path:
    explicit = record.get("target_stem_audio")
    if explicit:
        return Path(explicit)
    source = Path(record["source_audio"])
    stem = record.get("target_stem", "")
    candidates = [source.parent / "stems" / f"{stem}{extension}" for extension in (".wav", ".flac")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise KeyError("target_stem_audio is missing and no matching stem file was found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("data/derived.jsonl"), nargs="?")
    parser.add_argument("--example-id", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--limit", type=int, default=1, help="Number of labeled records to cycle through")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/latest"))
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    import soundfile as sf
    import torchaudio
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    record = next((item for item in records if item.get("example_id") == args.example_id), records[0])
    if args.instruction:
        record["instruction"] = args.instruction
    if args.example_id:
        records = [record]
    else:
        records = records[: max(1, args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_samples = int(32000 * args.duration_seconds)
    wave = load_audio(Path(record["source_audio"]), target_samples, sf, torchaudio)
    target_path = Path(record["target_audio"])
    if not target_path.exists():
        from build_edit_targets import render_target
        target_path = args.output_dir / "reference_target.wav"
        render_target(Path(record["source_audio"]), resolve_target_stem(record), target_path, record["operation"], Path(record["replacement_stem_audio"]) if record.get("replacement_stem_audio") else None)
    target_wave = load_audio(target_path, wave.shape[-1], sf, torchaudio)
    wave = wave[:, :target_samples]
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", local_files_only=True)
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", local_files_only=True)
    model = normalize_decoder_start_token(model)
    model.audio_encoder.eval()
    for parameter in model.audio_encoder.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05))
    model = normalize_decoder_start_token(model).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    examples = []
    for item in records:
        item_wave = load_audio(Path(item["source_audio"]), 32000 * 4, sf, torchaudio)
        item_target = Path(item["target_audio"])
        if not item_target.exists():
            from build_edit_targets import render_target
            item_target = args.output_dir / f"{item['example_id']}_target.wav"
            render_target(Path(item["source_audio"]), resolve_target_stem(item), item_target, item["operation"], Path(item["replacement_stem_audio"]) if item.get("replacement_stem_audio") else None)
        target = load_audio(item_target, item_wave.shape[-1], sf, torchaudio)
        item_inputs = processor(audio=item_wave.squeeze(0).numpy(), sampling_rate=32000, text=[item["instruction"]], return_tensors="pt")
        item_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in item_inputs.items()}
        target_inputs = processor(audio=target.squeeze(0).numpy(), sampling_rate=32000, text=[item["instruction"]], return_tensors="pt")
        target_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in target_inputs.items()}
        with torch.no_grad():
            codes = model.base_model.model.audio_encoder(**{key: target_inputs[key] for key in ("input_values", "padding_mask") if key in target_inputs}).audio_codes
        examples.append((item_inputs, codes[0].permute(0, 2, 1).contiguous()))
    losses = []
    model.train()
    for step in range(args.steps):
        inputs, labels = examples[step % len(examples)]
        optimizer.zero_grad()
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({"event": "progress", "step": len(losses), "steps": args.steps, "loss": losses[-1]}), flush=True)
    model.eval()
    print(json.dumps({"event": "phase", "phase": "generating"}), flush=True)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max(256, int(args.duration_seconds * 50)))
    source_path = args.output_dir / "source.wav"
    generated_path = args.output_dir / "generated_edit.wav"
    sf.write(source_path, wave.squeeze(0).detach().cpu().numpy(), 32000)
    generated_audio = generated[0].detach().float().cpu().numpy()
    if generated_audio.ndim > 1:
        generated_audio = generated_audio[0]
    # MusicGen returns the audio prompt followed by generated continuation.
    # Keep only the continuation and match the source duration for comparison.
    generated_audio = generated_audio[-wave.shape[-1]:]
    sf.write(generated_path, generated_audio, 32000, format="WAV", subtype="PCM_16")
    print(json.dumps({"device": str(device), "steps": args.steps, "examples": len(examples), "initial_loss": losses[0], "final_loss": losses[-1], "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "artifacts": {"source_audio": str(source_path), "generated_audio": str(generated_path)}, "status": "MusicGen labeled edit training completed"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
