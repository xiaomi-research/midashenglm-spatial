#!/usr/bin/env bash
#
# Run MMAU-Pro evaluation for MiDashengLM-Spatial. Mirrors scripts/run_starbench.sh:
# a MODE dispatcher over direct `python eval/mmaupro/*.py` calls, no conda switching.
#
#   MODE=standard  eval/mmaupro/infer.py  -> pred.jsonl + scored.parquet
#                  eval/mmaupro/eval.py   -> *_results.json (Qwen judge + NV-Embed + AIF)
#   MODE=lrswap    eval/mmaupro/test_lrswap.py, two stages across the two envs:
#                    stage 1 (INFER_PY): inference + regex report only (--no_nvembed)
#                    stage 2 (SCORE_PY): NV-Embed re-scoring, no inference (--rescore_only)
#   MODE=both      standard then lrswap
#
# Run from the repo root. Set CUDA_VISIBLE_DEVICES to pick the GPU. INFER_PY / SCORE_PY
# point at the model env and the NV-Embed env respectively (NV-Embed pins an older
# transformers, so they cannot share one env).
#
# Usage:
#   bash scripts/run_mmaupro.sh
#   MODE=both CUDA_VISIBLE_DEVICES=0 bash scripts/run_mmaupro.sh
#   MMAU_NSAMPLES=20 bash scripts/run_mmaupro.sh   # quick debug on first 20 rows

set -euo pipefail

# --- configuration (override via env) --------------------------------------
MODEL_PATH="${MODEL_PATH:-mispeech/midashenglm-spatial}"
BENCH_ROOT="${BENCH_ROOT:-Benchmarks/MMAU-Pro}"

MODE="${MODE:-both}"      # standard | lrswap | both
OUT_DIR="${OUT_DIR:-results/mmaupro}"

INFER_PY="${INFER_PY:-python}"
SCORE_PY="${SCORE_PY:-python}"


echo "=== MMAU-Pro: model=${MODEL_PATH} bench_root=${BENCH_ROOT} mode=${MODE} out_dir=${OUT_DIR} ==="

# validate mode early so a typo fails fast rather than after model load
case "${MODE}" in
    standard|lrswap|both) ;;
    *) echo "[err] MODE must be standard|lrswap|both (got '${MODE}')" >&2; exit 2 ;;
esac

# --- standard: full-parquet inference (+ scored parquet) then comprehensive scoring ---
run_standard() {
    local pred="${OUT_DIR}/standard/pred.jsonl"
    local scored="${OUT_DIR}/standard/scored.parquet"
    local infer_args=(--model_path "${MODEL_PATH}"
                      --parquet "${BENCH_ROOT}/test.parquet" --data_root "${BENCH_ROOT}"
                      --out_file "${pred}" --parquet_out "${scored}")

    echo "=== MMAU-Pro [standard] — inference + post-process (pred.jsonl + scored.parquet) ==="
    ${INFER_PY} eval/mmaupro/infer.py "${infer_args[@]}"

    echo "=== MMAU-Pro [standard] — scoring (Qwen judge + NV-Embed + AIF) ==="
    ${SCORE_PY} eval/mmaupro/eval.py "${scored}" \
        --model_output_column model_output --output_dir "${OUT_DIR}/standard"
}

# --- lrswap: L/R channel-swap directional robustness, split across the two envs ---------
#   stage 1 (INFER_PY):  inference + regex report only, NO NV-Embed (--no_nvembed)
#   stage 2 (SCORE_PY):  no inference, NV-Embed re-scoring of stage-1 runs (--rescore_only)
run_lrswap() {
    local runs="${OUT_DIR}/lrswap/lrswap_runs.jsonl"

    echo "=== MMAU-Pro [lrswap] — stage 1: inference + regex scoring (no NV-Embed) ==="
    ${INFER_PY} eval/mmaupro/test_lrswap.py \
        --model_path "${MODEL_PATH}" \
        --parquet "${BENCH_ROOT}/test.parquet" --data_root "${BENCH_ROOT}" \
        --out_file "${runs}" --no_nvembed --force

    echo "=== MMAU-Pro [lrswap] — stage 2: NV-Embed re-scoring (no inference) ==="
    ${SCORE_PY} eval/mmaupro/test_lrswap.py \
        --rescore_only --out_file "${runs}" --force
}

# --- dispatch --------------------------------------------------------------
case "${MODE}" in
    standard) run_standard ;;
    lrswap)   run_lrswap ;;
    both)     run_standard; run_lrswap ;;
esac

echo "=== done. results under ${OUT_DIR} ==="
