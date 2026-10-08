#!/usr/bin/env python3
"""Score STAR-Bench inference results (sr subset only): parse MCQ answers, compute AA & ACR.

Reads an `infer_results.jsonl` produced by `infer.py`, re-parses each model response
into a choice letter (following the official STAR-Bench evaluation code's
`parse_multi_choice_response`), marks correctness against the stored `answer_letter`,
and writes `performance.json` with:
  - AA  (Average Accuracy): mean of per-run correctness.
  - ACR (All-Correct Rate): fraction of unique samples correct on ALL rotation runs.
broken down by category and sub-category (plain micro AA/ACR).

Usage:
  python eval/starbench/eval.py <infer_results.jsonl> [--out performance.json]

"""

import argparse
import json
import os
import re
import string
import sys

import pandas as pd

UPPER = string.ascii_uppercase


def parse_multi_choice_response(response: str, options: list) -> str:
    """Extract predicted letter as '<A>'/'<B>'/... or 'Z' if no match.

    Vendored verbatim (logic) from the reference starbench.py.
    """
    response = str(response)
    letter2options = {UPPER[i]: options[i] for i in range(len(options))}
    all_choices = list(letter2options.keys())
    choices_str = "".join(all_choices)

    # Priority 1: choice mentioned after a keyword.
    keyword_matches = []
    keyword_pattern = re.compile(
        r"(?i)(?:answer|<answer>|solution|choice|option|correct option is|answer is|答案是|选项是)\s*[:：]*\s*"
        r"(?:"
        r"[\(\<\[\{{]\s*([{choices}])\s*[\)\]\>\}}]"
        r"|\b([{choices}])\b"
        r")".format(choices=choices_str)
    )
    for match in re.finditer(keyword_pattern, response):
        first, second = match.group(1), match.group(2)
        if second:
            keyword_matches.append((second, match.start(), 0))
        if first:
            keyword_matches.append((first, match.start(), 1))
    if keyword_matches:
        keyword_matches.sort(key=lambda x: (x[2], x[1]))
        return "<" + keyword_matches[-1][0] + ">"

    # Priority 2: any bracketed/standalone letter or full option-text match.
    standalone_matches = []
    standalone_pattern = re.compile(
        r"[\(\<\[\{{]\s*([{choices}])\s*[\)\]\>\}}]"
        r"|\b([{choices}])\b".format(choices=choices_str)
    )
    for match in re.finditer(standalone_pattern, response):
        first, second = match.group(1), match.group(2)
        if first:
            standalone_matches.append(("<" + first + ">", match.start(), 2))
        if second:
            standalone_matches.append(("<" + second + ">", match.start(), 0))
    if len(response.split()) > 2:
        for letter, text in letter2options.items():
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(text)}(?![A-Za-z])", re.IGNORECASE)
            for match in re.finditer(pattern, response):
                standalone_matches.append(("<" + letter + ">", match.start(), 1))
    if standalone_matches:
        standalone_matches.sort(key=lambda x: (x[2], x[1]))
        return standalone_matches[-1][0]

    return "Z"


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"AA": "0.00%", "ACR": "0.00%", "total_runs": 0, "unique_samples": 0}
    aa = df["is_correct"].mean() * 100
    acr = df.groupby("id")["is_correct"].all().mean() * 100
    return {
        "AA": f"{aa:.2f}%",
        "ACR": f"{acr:.2f}%",
        "total_runs": int(len(df)),
        "unique_samples": int(df["id"].nunique()),
    }


def evaluate_micro(df: pd.DataFrame) -> dict:
    """Plain micro AA/ACR with category / sub-category breakdown (sr)."""
    results = {"Total": _metrics(df)}
    if "category" in df.columns and df["category"].notna().any():
        results["By Category"] = {str(c): _metrics(g) for c, g in df.groupby("category")}
    if "sub-category" in df.columns and df["sub-category"].notna().any():
        results["By Sub-category"] = {str(s): _metrics(g) for s, g in df.groupby("sub-category")}
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Score STAR-Bench sr inference results (AA/ACR).")
    ap.add_argument("infer_results", help="infer_results.jsonl from infer.py.")
    ap.add_argument("--out", default=None, help="Output performance.json (default: alongside input).")
    args = ap.parse_args()

    if not os.path.exists(args.infer_results) or os.path.getsize(args.infer_results) == 0:
        print(f"[error] missing/empty: {args.infer_results}", file=sys.stderr)
        return 1

    rows = []
    with open(args.infer_results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pred = parse_multi_choice_response(item.get("model_response", ""), item["options"])
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category"),
                    "sub-category": item.get("sub-category"),
                    "prediction": pred,
                    "is_correct": pred == item["answer_letter"],
                }
            )
    df = pd.DataFrame(rows)
    print(f"[info] {len(df)} runs over {df['id'].nunique()} unique samples (subset=sr)")

    performance = evaluate_micro(df)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.infer_results)), "performance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(performance, f, indent=4, ensure_ascii=False)
    print(json.dumps(performance.get("Total", {}), indent=2, ensure_ascii=False))
    print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
