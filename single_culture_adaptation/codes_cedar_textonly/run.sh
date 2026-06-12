#!/usr/bin/env bash
set -euo pipefail

: "${CEDAR_TEXT_DATA_DIR:?Set CEDAR_TEXT_DATA_DIR to the CEDAR text-only multilingual data directory.}"

python run_benchmark_textonly.py \
  --data_dir "$CEDAR_TEXT_DATA_DIR" \
  --output_dir "${OUTPUT_DIR:-./results_single_culture}" \
  --mode framework_api \
  --base_url "${OPENAI_BASE_URL:-}" \
  --api_key "${OPENAI_API_KEY:-}" \
  --model_name "${MODEL_NAME:-gpt-4o}" \
  --max_workers "${MAX_WORKERS:-8}"
