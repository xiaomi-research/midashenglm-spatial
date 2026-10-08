"""MiDashengLM-Spatial inference on STAR-Bench (sr / holistic_reasoning_spatial subset only).

Usage:
    python eval/starbench/infer.py \
        --model_path <hf_repo_or_local_dir> \
        --dataset_root <STAR-Bench_root> \
        --out_dir <output_dir> \
        [--robust_eval True] [--n_samples N] [--max_length 1024] [--force]

Key args:
    --model_path    MiDashengLM-Spatial HF repo id or local model dir.
    --dataset_root  STAR-Bench root dir (contains meta_info/).
    --out_dir       Output dir; results written to <out_dir>/sr/infer_results.jsonl.
    --robust_eval   True/False; enable cyclic option-rotation robustness (3 runs/item).

"""

import argparse
import json
import logging
import os
import string
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.utils import generate, load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_starbench")

UPPER = string.ascii_uppercase

SUBSET = "sr"
SUBSET_JSON = "meta_info/holistic_reasoning_spatial.json"


def options_block(options):
    return "".join(f"<{UPPER[i]}>: {opt}\n" for i, opt in enumerate(options))


def build_runs(item, dataset_root, n_runs):
    """Build rotation runs for sr: cyclic option rotation, audio order fixed."""
    raw_options = item["options"]
    answer = item["answer"]
    audio_abs = [os.path.join(dataset_root, p) for p in item["audio_paths"]]
    multi = len(audio_abs) > 1

    runs = []
    for r in range(n_runs):
        prepared = raw_options[-r:] + raw_options[:-r] if r else list(raw_options)
        ans_idx = prepared.index(answer)
        answer_letter = f"<{UPPER[ans_idx]}>"

        prefix = ""
        if multi:
            prefix = "You will hear %d audio clips in order.\n" % len(audio_abs)
        text = f"{prefix}{item['question']}\n{options_block(prepared)}"

        runs.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "sub-category": item.get("sub-category"),
                "options": prepared,
                "answer_letter": answer_letter,
                "rotate_id": r,
                "prompt": text,
                "_audio": audio_abs,
            }
        )
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description="MiDashengLM-Spatial inference on STAR-Bench (sr subset).")
    ap.add_argument("--model_path", required=True, help="MiDashengLM-Spatial HF repo id or local model dir.")
    ap.add_argument("--dataset_root", required=True, help="STAR-Bench root dir (contains meta_info/).")
    ap.add_argument("--out_dir", required=True, help="Output dir; results in <out_dir>/sr/.")
    ap.add_argument("--robust_eval", default="True", help="True/False; enable option-rotation robustness.")
    ap.add_argument("--n_samples", type=int, default=None, help="Limit to first N items (debug).")
    ap.add_argument("--max_length", type=int, default=1024, help="Max new tokens to generate.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing infer_results.jsonl.")
    args = ap.parse_args()

    robust = str(args.robust_eval).lower() in ("1", "true", "yes")
    logger.info("Subset: %s | robust_eval=%s", SUBSET, robust)

    json_path = os.path.join(args.dataset_root, SUBSET_JSON)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.n_samples is not None:
        data = data[: args.n_samples]

    sub_dir = os.path.join(args.out_dir, SUBSET)
    os.makedirs(sub_dir, exist_ok=True)
    out_file = os.path.join(sub_dir, "infer_results.jsonl")
    if not args.force and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        logger.info("[%s] exists; skipping (use --force): %s", SUBSET, out_file)
        return 0

    model, processor, device = load_model(args.model_path)

    # Expand every item into its rotation runs (3 runs/item when robust, else 1).
    n_runs = 3 if robust else 1
    runs = []
    for item in data:
        runs.extend(build_runs(item, args.dataset_root, n_runs))
    logger.info("[%s] %d items -> %d runs", SUBSET, len(data), len(runs))

    results = []
    for r in tqdm(runs, desc=f"STAR-Bench[{SUBSET}]"):
        # `_audio` is a list of clip paths; load_audio concatenates them in order.
        pred = generate(
            r["_audio"], r["prompt"], model, processor, device,
            max_new_tokens=args.max_length,
        )
        rec = {k: v for k, v in r.items() if k != "_audio"}
        rec["unique_id"] = f"{r['id']}@{r['rotate_id']}"
        rec["model_response"] = pred
        results.append(rec)

    with open(out_file, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("[%s] wrote %d runs to %s", SUBSET, len(results), out_file)

    return 0


if __name__ == "__main__":
    code = main()
    raise SystemExit(code)
