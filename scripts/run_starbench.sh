#!/usr/bin/env bash
#
# Run STAR-Bench (sr / spatial subset) evaluation for MiDashengLM-Spatial. 
#
#   MODE=standard  eval/starbench/infer.py -> sr/infer_results.jsonl
#                  eval/starbench/eval.py  -> sr/performance.json (AA / ACR)
#   MODE=lrswap    eval/starbench/test_lrswap.py -> lrswap_runs.jsonl + .report.json
#                  (L/R channel-swap directional robustness; self-scoring)
#   MODE=both      standard then lrswap
#
# Run from the repo root. Set CUDA_VISIBLE_DEVICES to pick the GPU. Inference is
# single-GPU, single-thread.
#
# Usage:
#   BENCH_ROOT=/path/to/STAR-Bench bash scripts/run_starbench.sh
#   MODE=both CUDA_VISIBLE_DEVICES=0 BENCH_ROOT=/path/to/STAR-Bench bash scripts/run_starbench.sh

set -euo pipefail

# --- configuration (override via env) --------------------------------------
MODEL_PATH="${MODEL_PATH:-mispeech/midashenglm-spatial}"
BENCH_ROOT="${BENCH_ROOT:-Benchmarks/STAR-Bench}"

MODE="${MODE:-both}"        # standard | lrswap | both
OUT_DIR="${OUT_DIR:-results/starbench}"
ROBUST="${ROBUST:-True}"        # standard-mode option-rotation robustness (3 runs/item)

echo "=== STAR-Bench: model=${MODEL_PATH} BENCH_ROOT=${BENCH_ROOT} mode=${MODE} out_dir=${OUT_DIR} ==="

# validate mode early so a typo fails fast rather than after model load
case "${MODE}" in
    standard|lrswap|both) ;;
    *) echo "[err] MODE must be standard|lrswap|both (got '${MODE}')" >&2; exit 2 ;;
esac

# --- standard: option-rotation inference then AA/ACR scoring ----------------
run_standard() {
    local sub="${OUT_DIR}/standard"
    local infer_args=(--model_path "${MODEL_PATH}" --dataset_root "${BENCH_ROOT}"
                      --out_dir "${sub}" --robust_eval "${ROBUST}" --force)

    echo "=== STAR-Bench [standard] — inference (robust=${ROBUST}) ==="
    python eval/starbench/infer.py "${infer_args[@]}"

    local f="${sub}/sr/infer_results.jsonl"
    local p="${sub}/sr/performance.json"
    echo "=== STAR-Bench [standard] — scoring (AA / ACR) ==="
    if [[ -f "${f}" ]]; then
        python eval/starbench/eval.py "${f}" --out "${p}"
        [[ -f "${p}" ]] && echo "  sr Total: $(python -c "import json; print(json.load(open('${p}')).get('Total'))")"
    else
        echo "[warn] missing ${f}; nothing to score" >&2
    fi
}

# --- lrswap: L/R channel-swap directional robustness (self-scoring) ---------
run_lrswap() {
    local sub="${OUT_DIR}/lrswap"
    local args=(--model_path "${MODEL_PATH}" --dataset_root "${BENCH_ROOT}" --out_dir "${sub}" --force)

    echo "=== STAR-Bench [lrswap] — inference + scoring (L/R channel swap) ==="
    python eval/starbench/test_lrswap.py "${args[@]}"
}

# --- dispatch --------------------------------------------------------------
case "${MODE}" in
    standard) run_standard ;;
    lrswap)   run_lrswap ;;
    both)     run_standard; run_lrswap ;;
esac

echo "=== done. results under ${OUT_DIR} ==="
