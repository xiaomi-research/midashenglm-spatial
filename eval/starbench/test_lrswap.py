#!/usr/bin/env python3
"""L/R channel-swap directional-robustness eval on STAR-Bench spatial (sr) subset.

Probes spatial lateralization: each in-scope MCQ (Single-Source Static Localization) is
run as a paired trial with a FIXED option order -- run 0 = original audio + answer, run 1
= L/R-swapped audio (ch0<->ch1) + L/R-mirrored answer. Scoring reuses eval.py's
parse_multi_choice_response. Two kinds of items are excluded: direction-invariant ones
(the mirror leaves the answer unchanged, e.g. front/ahead/Yes-No) are dropped silently,
while items whose mirrored answer is not a verbatim option are logged to <out>.skipped.jsonl.

Usage:
    python eval/starbench/test_lrswap.py \
        --model_path <hf_repo_or_local_dir> \
        --dataset_root <STAR-Bench_root> \
        --out_dir <output_dir> \
        [--n_samples N] [--max_length 1024] [--force]

Key args:
    --model_path    MiDashengLM-Spatial HF repo id or local model dir.
    --dataset_root  STAR-Bench root dir (contains meta_info/).
    --out_dir       Output dir: lrswap_runs.jsonl + .report.json + .skipped.jsonl.
    --n_samples     Limit to first N meta items before scope filtering (debug).

"""

import argparse
import json
import logging
import os
import re
import string
import sys

import numpy as np
from tqdm import tqdm

# Make the repo root importable so `eval.*` resolves regardless of CWD.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.utils import generate, load_audio, load_model  # noqa: E402
# Reuse STAR-Bench's official letter parser so scores are comparable with the standard
# (option-rotation) STAR-Bench pipeline.
from eval.starbench.eval import parse_multi_choice_response  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_starbench_lrswap")

UPPER = string.ascii_uppercase

SUBSET = "sr"
SUBSET_JSON = "meta_info/holistic_reasoning_spatial.json"

# Word-boundary left/right matcher (case-insensitive), used for the answer mirror.
_RE_LR = re.compile(r"\b(left|right)\b", re.IGNORECASE)


def _match_case(src: str, word: str) -> str:
    """Return `word` re-cased to match `src` (ALL-CAPS, Capitalized, or lower)."""
    if src.isupper():
        return word.upper()
    if src[:1].isupper():
        return word.capitalize()
    return word.lower()


def flip_lr(text: str) -> str:
    """Swap every whole-word `left`<->`right` in `text`, preserving each word's case."""
    def repl(m: "re.Match") -> str:
        w = m.group(0)
        return _match_case(w, "right" if w.lower() == "left" else "left")
    return _RE_LR.sub(repl, str(text))


def _norm(s: str) -> str:
    return str(s).strip().lower()


def locate_answer_idx(answer: str, options: list):
    """Return the index of `answer` within `options` (normalized match), or None."""
    na = _norm(answer)
    for i, c in enumerate(options):
        if _norm(c) == na:
            return i
    return None


def options_block(options: list) -> str:
    return "".join(f"<{UPPER[i]}>: {opt}\n" for i, opt in enumerate(options))


def build_prompt(item: dict, options: list) -> str:
    """STAR-Bench MCQ prompt with a fixed option order (mirrors infer_starbench)."""
    audio_paths = item.get("audio_paths") or []
    prefix = ""
    if len(audio_paths) > 1:
        prefix = "You will hear %d audio clips in order.\n" % len(audio_paths)
    return f"{prefix}{item['question']}\n{options_block(options)}"


def _pct(numer: int, denom: int) -> float:
    return round(100.0 * numer / denom, 2) if denom else 0.0


def aggregate(runs: list) -> dict:
    """AA/ACR plus per-run accuracy (n_runs=2: run_0=orig, run_1=swapped).

    `runs` items carry: id, category, run_id (0=orig, 1=swapped), answer_letter,
    prediction, is_correct. run 0 and run 1 share the same option ORDER, so the parsed
    letter is directly comparable across the two runs of an item; comparing run_0 vs
    run_1 accuracy shows the accuracy change induced by the L/R channel swap.
    """
    n_runs = 2

    def metrics(items):
        if not items:
            return {
                "per_run": {}, "AA": 0.0, "ACR": 0.0,
                "total_runs": 0, "unique_samples": 0,
            }
        per_run = {}
        for rid in range(n_runs):
            sub = [r for r in items if r["run_id"] == rid]
            if sub:
                per_run[f"run_{rid}"] = {
                    "accuracy": _pct(sum(r["is_correct"] for r in sub), len(sub)),
                    "n": len(sub),
                }

        by_key = {}
        for r in items:
            by_key.setdefault(r["id"], []).append(r)

        all_correct = sum(1 for v in by_key.values() if all(r["is_correct"] for r in v))

        return {
            "per_run": per_run,
            "AA": _pct(sum(r["is_correct"] for r in items), len(items)),
            "ACR": _pct(all_correct, len(by_key)),          # == both_correct_rate
            "total_runs": len(items),
            "unique_samples": len(by_key),
        }

    report = {"Total": metrics(runs)}
    cats = sorted({r["category"] for r in runs if r.get("category") is not None})
    if cats:
        report["By Category"] = {c: metrics([r for r in runs if r["category"] == c]) for c in cats}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="L/R channel-swap STAR-Bench spatial eval.")
    ap.add_argument("--model_path", required=True, help="MiDashengLM-Spatial HF repo id or local model dir.")
    ap.add_argument("--dataset_root", required=True, help="STAR-Bench root dir (contains meta_info/).")
    ap.add_argument("--out_dir", required=True, help="Output dir (runs + report + skipped inside).")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="Limit to first N meta items before scope filtering (debug).")
    ap.add_argument("--max_length", type=int, default=1024, help="Max new tokens to generate.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing lrswap_runs.jsonl.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "lrswap_runs.jsonl")
    if not args.force and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        logger.info("Output exists and is non-empty; skipping (use --force): %s", out_file)
        return 0

    json_path = os.path.join(args.dataset_root, SUBSET_JSON)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = [d for d in data if d.get("category") == "Single-Source Static Localization"]
    if args.n_samples is not None:
        data = data[: args.n_samples]

    # Build paired (orig / swapped) records; collect skips for items whose mirrored answer
    # is not a verbatim option (unscoreable after the swap).
    records = []
    skipped = []
    for item in data:
        options = list(item["options"])
        answer = item["answer"]
        flipped_answer = flip_lr(answer)
        orig_idx = locate_answer_idx(answer, options)
        flip_idx = locate_answer_idx(flipped_answer, options)

        if flipped_answer == answer:
            # Direction-free / invariant answer (front/ahead/unchanged, Yes/No, object
            # name, ...): the L/R swap does not change the ground truth -> out of scope.
            continue
        if orig_idx is None or flip_idx is None:
            skipped.append({
                "id": item["id"],
                "category": item.get("category"),
                "question": item["question"],
                "answer": answer,
                "flipped_answer": flipped_answer,
                "options": options,
                "reason": ("orig_answer_not_in_options" if orig_idx is None
                           else "mirrored_answer_not_in_options"),
            })
            continue

        audio_abs = [os.path.join(args.dataset_root, p) for p in item["audio_paths"]]
        prompt = build_prompt(item, options)
        # run 0: original audio + original answer; run 1: swapped audio + mirrored answer.
        for run_id, (ans_idx, swapped) in enumerate([(orig_idx, False), (flip_idx, True)]):
            records.append({
                "id": item["id"],
                "category": item.get("category"),
                "sub-category": item.get("sub-category"),
                "run_id": run_id,
                "options": options,
                "answer_letter": f"<{UPPER[ans_idx]}>",
                "swapped": swapped,
                "orig_answer": answer,
                "flipped_answer": flipped_answer,
                "prompt": prompt,
                "_audio": audio_abs,
            })

    logger.info(
        "L/R-swap eval[%s]: %d meta items -> %d scoreable items (x2 runs = %d records), %d skipped.",
        SUBSET, len(data), len(records) // 2, len(records), len(skipped),
    )
    if not records:
        raise SystemExit("No scoreable directional items found; nothing to run.")

    model, processor, device = load_model(args.model_path)

    # Single-entry audio cache keyed by paths only: an item's two runs share the same decode
    # and differ only by channel order, so cache the raw wav and flip in memory on demand.
    cache = {"key": None, "wav": None}

    def load_audio_cached(paths, swapped):
        k = tuple(paths)
        if cache["key"] != k:
            cache["key"] = k
            cache["wav"] = load_audio(paths)  # (NUM_CHANNELS, n_samples) float32 array
        wav = cache["wav"]
        # ch0<->ch1 == left<->right; copy() to drop the negative stride so the
        # processor can wrap it with torch.from_numpy.
        return np.flip(wav, axis=0).copy() if swapped else wav

    runs = []
    for r in tqdm(records, desc=f"STAR-Bench L/R-swap[{SUBSET}]"):
        audio = load_audio_cached(r["_audio"], r["swapped"])
        pred = generate(
            audio, r["prompt"], model, processor, device,
            max_new_tokens=args.max_length,
        )
        pred_letter = parse_multi_choice_response(pred, r["options"])
        rec = {k: v for k, v in r.items() if k != "_audio"}
        rec["unique_id"] = f"{r['id']}@{r['run_id']}"
        rec["model_response"] = pred
        rec["prediction"] = pred_letter
        rec["is_correct"] = pred_letter == r["answer_letter"]
        runs.append(rec)

    with open(out_file, "w", encoding="utf-8") as w:
        for r in runs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d runs to %s", len(runs), out_file)

    skipped_file = out_file + ".skipped.jsonl"
    with open(skipped_file, "w", encoding="utf-8") as w:
        for r in skipped:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d skipped items to %s", len(skipped), skipped_file)

    report = aggregate(runs)
    report["config"] = {
        "n_runs": 2,
        "mode": "lr_channel_swap",  # run 0 = original, run 1 = L/R-swapped audio + mirrored answer
        "subset": SUBSET,
        "n_scoreable_items": len(records) // 2,
        "n_skipped": len(skipped),
        "scoring": "regex",  # parse_multi_choice_response (STAR-Bench official parser)
    }
    report_file = out_file + ".report.json"
    with open(report_file, "w", encoding="utf-8") as w:
        json.dump(report, w, indent=2, ensure_ascii=False)

    logger.info("L/R-swap report (%s):\n%s", report_file,
                json.dumps(report["Total"], indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
