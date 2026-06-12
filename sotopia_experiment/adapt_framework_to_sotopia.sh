#!/usr/bin/env bash
set -euo pipefail

export SOTOPIA_STORAGE_BACKEND="${SOTOPIA_STORAGE_BACKEND:-local}"

python "$(dirname "$0")/run_official_sotopia_framework.py" run \
  --models "${FRAMEWORK_MODEL_NAME:-gpt-4o}" \
  --partner-model "${SOTOPIA_PARTNER_MODEL:-together_ai/meta-llama/Llama-3-70b-chat-hf}" \
  --evaluator-model "${SOTOPIA_EVALUATOR_MODEL:-gpt-4o}" \
  --task "${SOTOPIA_TASK:-hard}" \
  --tag "${SOTOPIA_TAG:-framework_official_sotopia}" \
  --batch-size "${SOTOPIA_BATCH_SIZE:-10}"
