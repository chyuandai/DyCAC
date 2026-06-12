# Benchmark + Cultural Adaptive Multi-Agent Framework

This package integrates the uploaded cultural adaptive multi-agent framework into the uploaded benchmark with **minimal changes**:

- The benchmark's **data loading**, **prompt construction**, **response recording**, **answer extraction**, **validity judgment**, **accuracy computation**, **checkpointing**, and **result export** remain unchanged.
- The only integration change is at the model interface layer: the benchmark can now call the uploaded framework through a new `framework_api` mode.
- Each benchmark sample is run as an **independent single-turn interaction** through the framework, preserving the original benchmark assumption that `model.generate(prompt)` is stateless across samples.

## Files

- `run_benchmark_textonly.py` — modified benchmark runner
- `framework_benchmark_adapter.py` — adapter that invokes the uploaded framework pipeline per benchmark sample
- `llm_client.py`
- `perception.py`
- `memory_update.py`
- `cultural_hypothesis.py`
- `planning_execution.py`
- `main.py`

## New mode

`run_benchmark_textonly.py` now supports:

- `--mode vllm`
- `--mode api`
- `--mode framework_api`

## Example usage

```bash
python run_benchmark_textonly.py \
  --mode framework_api \
  --data_dir ./text-only/multilingual \
  --output_dir ./results \
  --base_url "$OPENAI_BASE_URL" \
  --api_key "$OPENAI_API_KEY" \
  --model_name gpt-4o
```

Optional framework parameters:

```bash
  --temp_perception 0.2 \
  --temp_memory 0.1 \
  --temp_culture 0.3 \
  --temp_response 0.7 \
  --n_hypotheses 5 \
  --framework_debug
```

## Integration principle

The framework itself has not been approximated or reimplemented with alternative logic. The uploaded framework modules are preserved and invoked through `run_pipeline(...)`; the benchmark continues to judge outputs exactly as before.
