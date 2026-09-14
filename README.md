# Tibetan FormatEval Benchmark — public 1,418-sample snapshot

This directory contains four aligned JSONL evaluation formats. Each file has
1,418 records in the same order and uses the same `id` values.

The public records retain dataset, subject, coarse domain, question, options,
answer indices, gold answer text, and open-answer aliases. Internal source
lineage and sample-split fields are intentionally omitted.

Files:

- `main.jsonl`: combined 4-way and 10-way representation.
- `mcqa_4way.jsonl`: four-choice multiple-choice evaluation format.
- `mcqa_10way.jsonl`: ten-choice multiple-choice evaluation format.
- `open_short_answer.jsonl`: open-answer evaluation format.

## Evaluation scripts

The `scripts/` directory contains standard-library-only Python 3.10+ tools for
running an OpenAI-compatible model endpoint, scoring one model's predictions,
and summarizing format-validity results across models.

### Generate predictions

`run_openai_compatible_eval.py` accepts either the 4-way or 10-way JSONL file
with `--task mcqa`, or the open-answer file with `--task open`:

```bash
python scripts/run_openai_compatible_eval.py \
  --gold mcqa_4way.jsonl \
  --task mcqa \
  --model YOUR_MODEL_NAME \
  --base-url https://YOUR_ENDPOINT/v1 \
  --api-key "$API_KEY" \
  --output predictions/YOUR_MODEL_NAME/mcqa_4way.jsonl \
  --concurrency 1
```

The runner writes resumable prediction JSONL records with the schema
`{"id": "...", "prediction": "..."}` plus model and request metadata. Use
an environment variable or a private `--shell-file` for credentials; do not
commit keys to the repository.

### Score predictions

`evaluate_predictions.py` supports letter/index MCQA predictions and normalized
open short answers. It writes both a summary JSON and a matching
`.per_item.jsonl` file:

```bash
python scripts/evaluate_predictions.py \
  --gold mcqa_4way.jsonl \
  --predictions predictions/YOUR_MODEL_NAME/mcqa_4way.jsonl \
  --task mcqa4 \
  --output results/YOUR_MODEL_NAME/mcqa_4way.json

python scripts/evaluate_predictions.py \
  --gold mcqa_10way.jsonl \
  --predictions predictions/YOUR_MODEL_NAME/mcqa_10way.jsonl \
  --task mcqa10 \
  --output results/YOUR_MODEL_NAME/mcqa_10way.json

python scripts/evaluate_predictions.py \
  --gold open_short_answer.jsonl \
  --predictions predictions/YOUR_MODEL_NAME/open_short_answer.jsonl \
  --task open \
  --output results/YOUR_MODEL_NAME/open_short_answer.json
```

### Compare multiple models

After each model has the three scored summaries and per-item files under its
own result directory, run:

```bash
python scripts/analyze_format_validity.py \
  --result-root results \
  --output results/format_validity_summary.json
```
