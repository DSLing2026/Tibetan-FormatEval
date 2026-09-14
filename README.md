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