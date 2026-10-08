#!/usr/bin/env python3
"""L/R channel-swap directional-robustness eval on MMAU-Pro spatial_audio.

Probes spatial lateralization: each in-scope Choice question (ground-truth answer
references exactly one lateral direction `left` XOR `right`, and no option mentions a
non-lateral axis such as front/back or up/down) is run as a paired trial with a FIXED
option order -- run 0 = original audio + answer, run 1 = L/R-swapped audio (ch0<->ch1)
+ L/R-mirrored answer. Items whose mirrored answer is not a verbatim option are dropped
to <out>.skipped.jsonl.

Scoring is NV-Embed-only (cosine argmax over option text; the letter-free prompt elicits
free-form answers regex can't parse). Because NV-Embed pins an older transformers than the
model, inference and scoring can run in separate envs:
  --no_nvembed    stage 1: inference only, write the runs JSONL (model env).
  --rescore_only  stage 2: NV-Embed re-score an existing runs JSONL (NV-Embed env).
A single-process run (neither flag) works when both fit one env (NV-Embed loads after the
model is freed).

Usage:
    python eval/mmaupro/test_lrswap.py \
        --model_path <hf_repo_or_local_dir> \
        --parquet <MMAU-Pro/test.parquet> \
        --data_root <MMAU-Pro_root> \
        --out_file <output_dir>/lrswap_runs.jsonl \
        [--category spatial_audio] [--n_samples N] [--max_length 1024] [--force]

Key args:
    --model_path    MiDashengLM-Spatial HF repo id or local model dir (required unless --rescore_only).
    --parquet       MMAU-Pro test.parquet (required unless --rescore_only).
    --data_root     Root dir the parquet's relative audio_path entries resolve against (required unless --rescore_only).
    --out_file      Output runs JSONL (.report_nvembed.json / .skipped.jsonl / .nvembed.jsonl written alongside).
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

# Make the repo root importable so `eval.*` resolves regardless of CWD.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.utils import generate, load_audio, load_model  # noqa: E402
# Reuse the (letter-tolerant) prompt builder so prompts stay identical to the standard
# inference pipeline. Scoring is NV-Embed-only (no regex parse).
from eval.mmaupro.infer import build_choice_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_lrswap")

NON_CHOICE = {"open", "instruction following"}

# Word-boundary left/right matchers (case-insensitive).
_RE_LEFT = re.compile(r"\bleft\b", re.IGNORECASE)
_RE_RIGHT = re.compile(r"\bright\b", re.IGNORECASE)
_RE_LR = re.compile(r"\b(left|right)\b", re.IGNORECASE)

# Non-lateral direction words (front/back and up/down axes). The L/R swap only mirrors
# left<->right, so any option referencing these axes is out of scope for a pure
# lateralization probe.
_RE_OTHER_DIRECTION = re.compile(
    r"\b("
    r"front|back|behind|rear|ahead|forward|frontal|posterior|anterior|"
    r"up|down|upper|lower|above|below|top|bottom|overhead|underneath|beneath|"
    r"north|south|east|west"
    r")\b",
    re.IGNORECASE,
)


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


def has_single_direction(answer: str) -> bool:
    """True iff `answer` mentions exactly one lateral direction (left XOR right)."""
    a = str(answer)
    return bool(_RE_LEFT.search(a)) ^ bool(_RE_RIGHT.search(a))


def choices_lateral_only(choices: list) -> bool:
    """True iff NO option references a non-lateral direction (front/back, up/down, ...)."""
    return not any(_RE_OTHER_DIRECTION.search(str(c)) for c in choices)


def _norm(s: str) -> str:
    return str(s).strip().lower()


def locate_answer_idx(answer: str, choices: list):
    """Return the index of `answer` within `choices` (normalized match), or None."""
    na = _norm(answer)
    for i, c in enumerate(choices):
        if _norm(c) == na:
            return i
    return None


def _pct(numer: int, denom: int) -> float:
    return round(100.0 * numer / denom, 2) if denom else 0.0


def aggregate(runs: list) -> dict:
    """AA/ACR plus per-run accuracy (n_runs=2: run_0=orig, run_1=swapped).

    `runs` items carry: key, category, run_id (0=orig, 1=swapped), answer_idx, pred_idx,
    is_correct. run 0 and run 1 share the same option ORDER, so comparing run_0 vs run_1
    accuracy shows the accuracy change induced by the L/R channel swap. Consumes the
    pred_idx / is_correct written by the NV-Embed scoring pass.
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
            by_key.setdefault(r["key"], []).append(r)

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


# --------------------------------------------------------------------------------------
# NV-Embed semantic scoring: match each free-form response to the option whose text is
# most similar under NV-Embed-v2 (cosine argmax), then feed aggregate().
# --------------------------------------------------------------------------------------

def load_nvembed(model_name: str = "nvidia/NV-Embed-v2"):
    """Load NV-Embed-v2."""
    logger.info("Loading %s ...", model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    logger.info("%s loaded.", model_name)
    return model


def _encode(model, texts, max_length):
    """Encode a list of strings to L2-normalized embeddings (NV-Embed API)."""
    try:
        emb = model.encode(texts, instruction="", max_length=max_length)
    except Exception:  # some NV-Embed builds want a single string, not a 1-list
        emb = model.encode(texts[0] if len(texts) == 1 else texts, instruction="", max_length=max_length)
    if not torch.is_tensor(emb):
        emb = torch.as_tensor(emb)
    return F.normalize(emb, p=2, dim=1)


def nvembed_match_idx(model, response: str, choices: list, max_length: int) -> int:
    """Return the index of the option most similar to `response` under NV-Embed (argmax)."""
    resp_emb = _encode(model, [str(response) if response is not None else ""], max_length)
    choice_emb = _encode(model, [str(c) for c in choices], max_length)
    scores = (resp_emb @ choice_emb.T) * 100.0
    return int(torch.argmax(scores.squeeze(0)).item())


def nvembed_rescore(runs: list, model, max_length: int) -> list:
    """Score `runs` by NV-Embed option-text matching; set pred_idx / is_correct per run."""
    scored = []
    for r in tqdm(runs, desc="NV-Embed score"):
        choices = list(r["choices"])
        model_output = r.get("model_output", "")
        # A bare-letter answer ("A"/"b"/...) -> map to its option text before embedding.
        if isinstance(model_output, str) and len(model_output) == 1:
            li = ord(model_output.upper()) - 65
            if 0 <= li < len(choices):
                model_output = choices[li]
        with torch.no_grad():
            pred_idx = nvembed_match_idx(model, model_output, choices, max_length)
        s = dict(r)
        s["pred_idx"] = pred_idx
        s["model_output"] = model_output
        s["is_correct"] = r.get("answer_idx") is not None and pred_idx == r["answer_idx"]
        scored.append(s)
    return scored


def run_nvembed_pass(runs: list, args) -> None:
    """Run the NV-Embed scoring pass and write <out_file>.nvembed.jsonl + report."""
    scored_file = args.out_file + ".nvembed.jsonl"
    report_file = args.out_file + ".report_nvembed.json"
    if not args.force and os.path.exists(report_file) and os.path.getsize(report_file) > 0:
        logger.info("NV-Embed report exists and is non-empty; skipping (use --force): %s", report_file)
        return

    model = load_nvembed(args.nvembed_model)
    scored = nvembed_rescore(runs, model, args.nvembed_max_length)

    with open(scored_file, "w", encoding="utf-8") as w:
        for r in scored:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d NV-Embed-scored runs to %s", len(scored), scored_file)

    report = aggregate(scored)
    report["config"] = {
        "n_runs": 2,
        "mode": "lr_channel_swap",  # run 0 = original, run 1 = L/R-swapped audio + mirrored answer
        "category": args.category,
        "scoring": "nvembed",  # NV-Embed-v2 cosine argmax over option text (no regex parsing)
        "nvembed_model": args.nvembed_model,
        "n_scoreable_questions": len({r["key"] for r in scored}),
    }
    with open(report_file, "w", encoding="utf-8") as w:
        json.dump(report, w, indent=2, ensure_ascii=False)
    logger.info("NV-Embed L/R-swap report (%s):\n%s", report_file,
                json.dumps(report["Total"], indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="L/R channel-swap MMAU-Pro spatial eval (NV-Embed scored).")
    ap.add_argument("--model_path", default=None, help="MiDashengLM-Spatial HF repo id or local model dir (required unless --rescore_only).")
    ap.add_argument("--parquet", default=None, help="MMAU-Pro test.parquet (required unless --rescore_only).")
    ap.add_argument("--data_root", default=None, help="Root dir for relative audio_path entries (required unless --rescore_only).")
    ap.add_argument("--out_file", required=True, help="Output runs JSONL (report / skipped / nvembed written alongside).")
    ap.add_argument("--category", default="spatial_audio", help="Category to evaluate (default: spatial_audio).")
    ap.add_argument("--n_samples", type=int, default=None, help="Limit to first N rows (debug).")
    ap.add_argument("--max_length", type=int, default=1024, help="Max new tokens to generate.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing out_file / report.")
    ap.add_argument("--no_nvembed", action="store_true",
                    help="Inference stage only: write the runs JSONL, skip NV-Embed scoring (no report).")
    ap.add_argument("--rescore_only", action="store_true",
                    help="Skip inference; NV-Embed score an existing --out_file (no model env needed).")
    ap.add_argument("--nvembed_model", default="nvidia/NV-Embed-v2", help="NV-Embed model id (default: nvidia/NV-Embed-v2).")
    ap.add_argument("--nvembed_max_length", type=int, default=4096, help="NV-Embed max_length (default 4096).")
    args = ap.parse_args()

    # Mode A: score an existing runs file only (no inference).
    if args.rescore_only:
        if not os.path.exists(args.out_file) or os.path.getsize(args.out_file) == 0:
            raise SystemExit(f"--rescore_only needs an existing non-empty runs file: {args.out_file}")
        with open(args.out_file, "r", encoding="utf-8") as f:
            runs = [json.loads(line) for line in f if line.strip()]
        if not runs:
            raise SystemExit(f"No runs in {args.out_file}")
        logger.info("Loaded %d runs from %s for NV-Embed scoring.", len(runs), args.out_file)
        run_nvembed_pass(runs, args)
        return 0

    # Mode B: full pipeline (inference -> NV-Embed report).
    for name in ("model_path", "parquet", "data_root"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name} is required unless --rescore_only is set.")

    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        logger.info("Output exists and is non-empty; skipping inference (use --force): %s", args.out_file)
        if not args.no_nvembed:
            with open(args.out_file, "r", encoding="utf-8") as f:
                runs = [json.loads(line) for line in f if line.strip()]
            run_nvembed_pass(runs, args)
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    df = pd.read_parquet(args.parquet, engine="pyarrow")
    if args.category is not None:
        df = df[df["category"] == args.category].copy()
        if len(df) == 0:
            raise SystemExit(f"No rows for category={args.category!r} in {args.parquet}")
    if args.n_samples is not None:
        df = df.iloc[: args.n_samples].copy()

    # Keep only Choice questions with options.
    df = df[~df["category"].isin(NON_CHOICE)].copy()
    df = df[df["choices"].apply(lambda c: c is not None and len(c) > 1)].copy()

    # Build paired (orig / swapped) records; collect skips for questions whose mirrored
    # answer is not a verbatim option (unscoreable after the swap).
    records = []
    skipped = []
    for _, row in df.iterrows():
        answer = row["answer"]
        if not has_single_direction(answer):
            continue  # not a single-lateral-direction answer -> out of scope
        choices = list(row["choices"])
        flipped_answer = flip_lr(answer)
        # Exclude questions whose options reference non-lateral axes (front/back, up/down):
        # the L/R swap does not mirror these, so they would contaminate the probe.
        if not choices_lateral_only(choices):
            skipped.append({
                "key": row["id"], "category": row["category"], "question": row["question"],
                "answer": answer, "flipped_answer": flipped_answer, "choices": choices,
                "reason": "options_contain_non_lateral_direction",
            })
            continue
        orig_idx = locate_answer_idx(answer, choices)
        flip_idx = locate_answer_idx(flipped_answer, choices)
        if orig_idx is None or flip_idx is None:
            skipped.append({
                "key": row["id"], "category": row["category"], "question": row["question"],
                "answer": answer, "flipped_answer": flipped_answer, "choices": choices,
                "reason": ("orig_answer_not_in_choices" if orig_idx is None
                           else "mirrored_answer_not_in_choices"),
            })
            continue

        audio_rel = list(row["audio_path"]) if row["audio_path"] is not None else []
        audio_abs = [os.path.join(args.data_root, p) for p in audio_rel]
        prompt = build_choice_prompt(row["question"], choices)[0]
        # run 0: original audio + original answer; run 1: swapped audio + mirrored answer.
        for run_id, (ans, ans_idx, swapped) in enumerate(
            [(answer, orig_idx, False), (flipped_answer, flip_idx, True)]
        ):
            records.append({
                "key": row["id"], "category": row["category"], "question": row["question"],
                "run_id": run_id, "choices": choices,
                "answer": ans, "answer_idx": ans_idx,
                "orig_answer": answer, "flipped_answer": flipped_answer, "swapped": swapped,
                "prompt": prompt, "_audio": audio_abs,
            })

    logger.info(
        "L/R-swap eval: %d %s rows -> %d scoreable questions (x2 runs = %d records), %d skipped.",
        len(df), args.category, len(records) // 2, len(records), len(skipped),
    )
    if not records:
        raise SystemExit("No scoreable single-direction questions found; nothing to run.")

    model, processor, device = load_model(args.model_path)

    # Single-entry cache keyed by paths only: a question's two runs share the same decode
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
    for r in tqdm(records, desc="MMAU-Pro L/R-swap infer"):
        audio = load_audio_cached(r["_audio"], r["swapped"])
        p = generate(audio, r["prompt"], model, processor, device, max_new_tokens=args.max_length)
        # No regex scoring: the raw model_output is scored by the NV-Embed pass.
        rec = {k: v for k, v in r.items() if k != "_audio"}
        rec["model_output"] = p
        runs.append(rec)

    with open(args.out_file, "w", encoding="utf-8") as w:
        for r in runs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d runs to %s", len(runs), args.out_file)

    skipped_file = args.out_file + ".skipped.jsonl"
    with open(skipped_file, "w", encoding="utf-8") as w:
        for r in skipped:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d skipped questions to %s", len(skipped), skipped_file)

    # NV-Embed scoring pass. Free the MiDashengLM-Spatial model first to limit peak GPU
    # memory before loading NV-Embed-v2.
    if not args.no_nvembed:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_nvembed_pass(runs, args)

    return 0


if __name__ == "__main__":
    tic = time.time()
    code = main()
    logger.info("Done in %.1f min", (time.time() - tic) / 60)
    raise SystemExit(code)
