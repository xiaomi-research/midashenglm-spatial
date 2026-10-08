#!/usr/bin/env python3
"""Minimal inference example for MiDashengLM-Spatial.
Usage:
  python inference.py --model /path/to/midashenglm-spatial-7b --audio /path/to/binaural.wav
  python inference.py --model mispeech/midashenglm-spatial --audio a.wav \
      --prompt "Write a general audio caption describing the sound source and its spatial information."
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_MODEL = "mispeech/midashenglm-spatial"
DEFAULT_AUDIO = "example/example.wav"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local package dir")
    ap.add_argument("--audio", default=DEFAULT_AUDIO, help="path to a (mono or binaural) audio file")
    ap.add_argument("--prompt", default="Write a general audio caption describing the sound source and its spatial information.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": args.audio},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    with torch.no_grad():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
        ).to(device)
        generation = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        output = processor.tokenizer.batch_decode(generation, skip_special_tokens=True)

    print(f"[audio] {args.audio!r}")
    print(f"[prompt] {args.prompt!r}")
    print(f"[output] {output[0]}")


if __name__ == "__main__":
    main()
