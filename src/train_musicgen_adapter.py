#!/usr/bin/env python3
"""Minimal one-example MusicGen LoRA reconstruction smoke test."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("data/derived.jsonl"), nargs="?")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    args = parser.parse_args()
    import soundfile as sf
    import torchaudio
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    record = json.loads(args.manifest.read_text().splitlines()[0])
    audio, sample_rate = sf.read(record["source_audio"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wave = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != 32000:
        wave = torchaudio.functional.resample(wave, sample_rate, 32000)
    wave = wave[:, : 32000 * 4]
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", local_files_only=True)
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", local_files_only=True)
    model = normalize_decoder_start_token(model)
    model.audio_encoder.eval()
    for parameter in model.audio_encoder.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05))
    model = normalize_decoder_start_token(model).to(device)
    inputs = processor(audio=wave.squeeze(0).numpy(), sampling_rate=32000, text=[record["instruction"]], return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        codes = model.base_model.model.audio_encoder(**{key: inputs[key] for key in ("input_values", "padding_mask") if key in inputs}).audio_codes
    labels = codes[0].permute(0, 2, 1).contiguous()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    losses = []
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({"event": "progress", "step": len(losses), "steps": args.steps, "loss": losses[-1]}), flush=True)
    print(json.dumps({"device": str(device), "steps": args.steps, "initial_loss": losses[0], "final_loss": losses[-1], "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "status": "MusicGen LoRA reconstruction smoke test passed"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
