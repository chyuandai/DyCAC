## Run

```bash
cd <PROJECT_ROOT>
python run_benchmark_multimodal_framework.py \
  --json_folder /path/to/multimodal/multilingual \
  --image_folder /path/to/multimodal/images \
  --output_dir ./results \
  --mode framework_api \
  --base_url "$OPENAI_BASE_URL" \
  --api_key "$OPENAI_API_KEY" \
  --model_name gpt-4o
```

Optional framework args:
- `--temp_perception 0.2`
- `--temp_memory 0.1`
- `--temp_culture 0.3`
- `--temp_response 0.7`
- `--n_hypotheses 5`
- `--social_goal ""`
- `--agent_persona "a culturally aware conversational assistant"`
- `--debug_framework`

