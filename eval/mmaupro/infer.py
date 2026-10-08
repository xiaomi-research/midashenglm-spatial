#!/usr/bin/env python3
"""MiDashengLM-Spatial inference on MMAU-Pro (raw audio, no tar).

Reads MMAU-Pro test.parquet, builds the official per-category prompt (open / instruction
following / Choice), runs greedy generation over the referenced audio (STEREO via
eval.utils.load_audio), and writes a predictions JSONL (sample fields + model_output).

Optionally emits the scored parquet consumed by the comprehensive scorer: the reference
parquet with a model_output column where single-letter Choice answers (e.g. "B") are mapped
back to the full option text. Pass --parquet_out after inference, or --postprocess_only to
build it from an existing predictions JSONL (no model env needed).

Usage:
    python eval/mmaupro/infer.py \
        --model_path <hf_repo_or_local_dir> \
        --parquet <MMAU-Pro/test.parquet> \
        --data_root <MMAU-Pro_root> \
        --out_file <output_dir>/pred.jsonl \
        [--parquet_out <output_dir>/pred.parquet] [--n_samples N] [--max_length 1024] [--force]

Key args:
    --model_path       MiDashengLM-Spatial HF repo id or local model dir (required unless --postprocess_only).
    --parquet          MMAU-Pro test.parquet (also the --parquet_out reference unless --refer overrides).
    --data_root        Root dir the parquet's relative audio_path entries resolve against (required unless --postprocess_only).
    --out_file         Output predictions JSONL.
    --parquet_out      Optional scored parquet (reference + model_output) for the comprehensive scorer.
    --refer            Reference parquet for --parquet_out (default: --parquet).
    --postprocess_only Skip inference; build --parquet_out from an existing --out_file JSONL.
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import pandas as pd
from tqdm import tqdm

# Make the repo root importable so `eval.utils` resolves regardless of CWD.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.utils import generate, load_audio, load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_mmaupro")

CHOICE_LETTERS = [chr(65 + i) for i in range(26)]


def build_choice_prompt(question: str, choices) -> tuple:
    """Build a Choice-question prompt following the official MMAU-Pro format.

    Returns ``(prompt_text, "Choice")``.
    """
    opts = list(choices) if choices is not None else []
    lines = "\n".join(f"({CHOICE_LETTERS[i]}) {item}" for i, item in enumerate(opts))
    return question + "\n" + lines, "Choice"


def build_prompt(category: str, question: str, choices) -> tuple:
    """Return (prompt_text, task_tag) following the official MMAU-Pro format."""
    if category == "open":
        return question + " Answer with 1-2 sentences.", "Open"
    if category == "instruction following":
        return question, "Open"
    return build_choice_prompt(question, choices)


def build_scored_parquet(results: list, refer_parquet: str, out_parquet: str) -> None:
    """Merge predictions into the reference parquet for the comprehensive scorer.

    Merged in from the former standalone `post_mmaupro.py` (logic unchanged): for Choice
    questions where the model emitted a single option letter (e.g. "B"), map that letter
    back to the full option text; everything else passes through. Adds a `model_output`
    column to the reference parquet (matched by `id`) and writes it to `out_parquet`.
    """
    data = {}
    for temp in results:
        mo = temp["model_output"]
        if temp.get("task") == "Choice" and isinstance(mo, str) and len(mo) == 1:
            flag = ord(mo.upper()) - 65
            choices = temp.get("choices") or []
            if flag < 0 or flag >= len(choices):
                logger.warning("Choice letter out of range for key=%s: %r", temp["key"], mo)
                data[temp["key"]] = mo
            else:
                data[temp["key"]] = choices[flag]
        else:
            data[temp["key"]] = mo

    df = pd.read_parquet(refer_parquet, engine="pyarrow")
    df["model_output"] = None
    for i in df.index:
        df.at[i, "model_output"] = data[df.at[i, "id"]]
    os.makedirs(os.path.dirname(os.path.abspath(out_parquet)) or ".", exist_ok=True)
    df.to_parquet(out_parquet, engine="pyarrow", index=False)
    logger.info("Wrote scored parquet (%d rows) to %s", len(df), out_parquet)


def main() -> int:
    ap = argparse.ArgumentParser(description="MiDashengLM-Spatial inference on MMAU-Pro (+ scored-parquet post-processing).")
    ap.add_argument("--model_path", default=None, help="MiDashengLM-Spatial HF repo id or local model dir (required unless --postprocess_only).")
    ap.add_argument("--parquet", required=True, help="MMAU-Pro test.parquet (also the --parquet_out reference unless --refer overrides).")
    ap.add_argument("--data_root", default=None, help="Root dir for relative audio_path entries (required unless --postprocess_only).")
    ap.add_argument("--out_file", required=True, help="Output predictions JSONL.")
    ap.add_argument("--parquet_out", default=None, help="Optional scored parquet (reference + model_output) for the comprehensive scorer.")
    ap.add_argument("--refer", default=None, help="Reference parquet for --parquet_out (default: --parquet).")
    ap.add_argument("--postprocess_only", action="store_true",
                    help="Skip inference; build --parquet_out from an existing --out_file JSONL.")
    ap.add_argument("--n_samples", type=int, default=None, help="Limit to first N rows (debug).")
    ap.add_argument("--max_length", type=int, default=1024, help="Max new tokens to generate.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing out_file / parquet_out.")
    args = ap.parse_args()

    # Mode A: post-process an existing predictions JSONL into the scored parquet only.
    if args.postprocess_only:
        if not args.parquet_out:
            raise SystemExit("--postprocess_only requires --parquet_out.")
        if not os.path.exists(args.out_file) or os.path.getsize(args.out_file) == 0:
            raise SystemExit(f"--postprocess_only needs an existing non-empty JSONL: {args.out_file}")
        if not args.force and os.path.exists(args.parquet_out) and os.path.getsize(args.parquet_out) > 0:
            logger.info("parquet_out exists and is non-empty; skipping (use --force): %s", args.parquet_out)
            return 0
        with open(args.out_file, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d predictions from %s for post-processing.", len(results), args.out_file)
        build_scored_parquet(results, args.refer or args.parquet, args.parquet_out)
        return 0

    # Mode B: full inference (optionally also emitting the scored parquet).
    for name in ("model_path", "data_root"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name} is required unless --postprocess_only is set.")

    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        logger.info("Output exists and is non-empty; skipping inference (use --force): %s", args.out_file)
        # Still (re-)build the scored parquet from the existing JSONL if requested.
        if args.parquet_out:
            with open(args.out_file, "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f if line.strip()]
            build_scored_parquet(results, args.refer or args.parquet, args.parquet_out)
        return 0

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet, engine="pyarrow")
    if args.n_samples is not None:
        df = df.iloc[: args.n_samples].copy()
    logger.info("Loaded %d MMAU-Pro samples from %s", len(df), args.parquet)

    model, processor, device = load_model(args.model_path)

    # Pre-build per-row records (prompt + resolved audio paths + GT fields).
    records = []
    for _, row in df.iterrows():
        prompt, task = build_prompt(row["category"], row["question"], row.get("choices"))
        audio_rel = list(row["audio_path"]) if row["audio_path"] is not None else []
        audio_abs = [os.path.join(args.data_root, p) for p in audio_rel]
        choices = list(row["choices"]) if row["choices"] is not None else None
        records.append(
            {
                "key": row["id"],
                "category": row["category"],
                "question": row["question"],
                "answer": row["answer"],
                "choices": choices,
                "task": task,
                "prompt": prompt,
                "_audio": audio_abs,
            }
        )

    results = []
    for r in tqdm(records, desc="MMAU-Pro infer"):
        # `_audio` is a list of clip paths; load_audio concatenates them in order.
        pred = generate(
            r["_audio"], r["prompt"], model, processor, device,
            max_new_tokens=args.max_length,
        )
        out = {k: v for k, v in r.items() if k != "_audio"}
        out["model_output"] = pred
        results.append(out)

    with open(args.out_file, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d predictions to %s", len(results), args.out_file)

    # Optionally emit the scored parquet consumed by the comprehensive scorer.
    if args.parquet_out:
        build_scored_parquet(results, args.refer or args.parquet, args.parquet_out)
    return 0


if __name__ == "__main__":
    tic = time.time()
    code = main()
    logger.info("Done in %.1f min", (time.time() - tic) / 60)
    raise SystemExit(code)
