# DyCAC Supplementary Codes

This archive contains the supplementary code for **DyCAC**, a training-free framework with dynamic cultural adaptation and continuous cognitive tracking.

The code is organized to reproduce the main framework runs, SOTOPIA evaluation, CEDAR evaluations through the official benchmark wrapper, and ablation variants.

## Repository layout

```text
framework_codes/                         Core DyCAC pipeline
cedar_experiment/text-only/              CEDAR text-only evaluation 
cedar_experiment/multimodal/             CEDAR multimodal evaluation 
sotopia_experiment/                      Official SOTOPIA wrapper and launch script
ablation_study/                          Ablation variants
single_culture_adaptation/               Single-culture adaptation baseline variants
```

The core pipeline is split into four modules:

1. `perception.py` extracts objective facts, mental-state signals, and cultural cues.
2. `memory_update.py` maintains explicit dialogue memory and ToM-Driven cognitive state estimates.
3. `cultural_hypothesis.py` updates the dynamic culture space and produces soft cultural profiles of the interlocutor.
4. `planning_execution.py` plans and generates the final response using the current memory and cultural profile.

`main.py` orchestrates these modules through `run_pipeline(...)`.

## Environment

Common environment variables:

```bash
export OPENAI_API_KEY="<API_KEY>"
export OPENAI_BASE_URL="<BASE_URL>"        # optional; leave unset for the provider default
export MODEL_NAME="gpt-4o"                 # or any OpenAI-compatible model name
```

## Quick start: run the core framework

```bash
cd framework_codes
python main.py \
  --input "Hello, I am preparing for an important job interview and I feel nervous." \
  --api_key "$OPENAI_API_KEY" \
  --base_url "${OPENAI_BASE_URL:-}" \
  --model_name "${MODEL_NAME:-gpt-4o}" \
  --debug
```

For interactive multi-turn mode, omit `--input`:

```bash
python main.py \
  --api_key "$OPENAI_API_KEY" \
  --base_url "${OPENAI_BASE_URL:-}" \
  --model_name "${MODEL_NAME:-gpt-4o}"
```

Optional parameters include:

```text
--temp_perception   Temperature for the perception module, default 0.2
--temp_memory       Temperature for the memory update module, default 0.1
--temp_culture      Temperature for cultural profile inference, default 0.3
--temp_response     Temperature for planning and response generation, default 0.7
--n_hypotheses      Number of cultural profiles to maintain, default 5 or 10
--social_goal       Optional SOTOPIA-style goal for goal-directed interaction
--agent_persona     Optional role description for the agent (used in SOTOPIA evaluation)
```

## Run CEDAR text-only evaluation

Set the CEDAR text-only data directory and run:

```bash
cd cedar_experiment/text-only
export CEDAR_TEXT_DATA_DIR="<CEDAR_TEXT_ONLY_MULTILINGUAL_DIR>"
./run.sh
```

Equivalent explicit command:

```bash
python run_benchmark_textonly.py \
  --mode framework_api \
  --data_dir "$CEDAR_TEXT_DATA_DIR" \
  --output_dir ./results \
  --base_url "${OPENAI_BASE_URL:-}" \
  --api_key "$OPENAI_API_KEY" \
  --model_name "${MODEL_NAME:-gpt-4o}"
```

## Run CEDAR multimodal evaluation

Set the CEDAR multimodal JSON and image directories and run:

```bash
cd cedar_experiment/multimodal
export CEDAR_MULTIMODAL_JSON_DIR="<CEDAR_MULTIMODAL_JSON_DIR>"
export CEDAR_MULTIMODAL_IMAGE_DIR="<CEDAR_MULTIMODAL_IMAGE_DIR>"
./run.sh
```

Equivalent explicit command:

```bash
python run_benchmark_multimodal_framework.py \
  --mode framework_api \
  --json_folder "$CEDAR_MULTIMODAL_JSON_DIR" \
  --image_folder "$CEDAR_MULTIMODAL_IMAGE_DIR" \
  --output_dir ./results \
  --base_url "${OPENAI_BASE_URL:-}" \
  --api_key "$OPENAI_API_KEY" \
  --model_name "${MODEL_NAME:-gpt-4o}"
```

## Run SOTOPIA evaluation

The SOTOPIA adapter uses the official SOTOPIA benchmark implementation and injects the framework as a custom agent.

Note that please download the SOTOPIA datasets and codes and CEDAR datasets before the evaluation.

```bash
pip install sotopia
sotopia install

cd sotopia_experiment
export OPENAI_API_KEY="<API_KEY>"
export FRAMEWORK_API_KEY="$OPENAI_API_KEY"
export FRAMEWORK_BASE_URL="${OPENAI_BASE_URL:-}"
./run_official_framework_benchmark.sh
```

You can also call the wrapper directly:

```bash
python sotopia_experiment/adapt_framework_to_sotopia.py run \
  --models "${MODEL_NAME:-gpt-4o}" \
  --partner-model "<PARTNER_MODEL_NAME>" \
  --evaluator-model "<EVALUATOR_MODEL_NAME>" \
  --task hard \
  --tag framework_official_sotopia
```

## Ablation and single-culture variants

Ablation variants under `ablation_study/` preserve the same command-line interface as `framework_codes/main.py`. For example:

```bash
cd ablation_study/wo_full_memory
python main.py \
  --input "Your dialogue input here." \
  --api_key "$OPENAI_API_KEY" \
  --base_url "${OPENAI_BASE_URL:-}" \
  --model_name "${MODEL_NAME:-gpt-4o}"
```

Single-culture CEDAR variants under `single_culture_adaptation/` provide the same benchmark entry points as the corresponding CEDAR folders.

## Outputs

Benchmark scripts write result files to the selected `--output_dir`. The exact files depend on the benchmark mode and may include CSV, TXT, JSON checkpoint, and summary files.


