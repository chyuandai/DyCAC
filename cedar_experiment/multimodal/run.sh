#!/usr/bin/env bash
set -euo pipefail

: "${CEDAR_MULTIMODAL_JSON_DIR:?Set CEDAR_MULTIMODAL_JSON_DIR to the CEDAR multimodal multilingual JSON directory.}"
: "${CEDAR_MULTIMODAL_IMAGE_DIR:?Set CEDAR_MULTIMODAL_IMAGE_DIR to the CEDAR multimodal image directory.}"

python run_benchmark_multimodal_framework.py \
  --json_folder "$CEDAR_MULTIMODAL_JSON_DIR" \
  --image_folder "$CEDAR_MULTIMODAL_IMAGE_DIR" \
  --output_dir "${OUTPUT_DIR:-./results}" \
  --mode framework_api \
  --base_url "${OPENAI_BASE_URL:-}" \
  --api_key "${OPENAI_API_KEY:-}" \
  --model_name "${MODEL_NAME:-gpt-4o}"
