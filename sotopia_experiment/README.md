## Setup

```bash
pip install sotopia
sotopia install
export SOTOPIA_STORAGE_BACKEND=local
export OPENAI_API_KEY=...
export FRAMEWORK_API_KEY="$OPENAI_API_KEY"
```

`local_sotopia_data/` is only a placeholder. For real runs, initialize the official SOTOPIA data/backend through the official package.

`sotopia_repo/` is only a placeholder. For real runs, initialize the SOTOPIA evaluation through the official package.

## Run

```bash
python sotopia_experiment/adapt_framework_to_sotopia.py run \
  --models gpt-4o \
  --partner-model together_ai/meta-llama/Llama-3-70b-chat-hf \
  --evaluator-model gpt-4o \
  --task hard \
  --tag framework_official_sotopia
```

## Display existing results

```bash
python sotopia_experiment/adapt_framework_to_sotopia.py display \
  --models gpt-4o \
  --tag framework_official_sotopia
```

