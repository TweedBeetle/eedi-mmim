# Eedi MMIM re-evaluation, September 2026

This directory holds a held-out evaluation of the October 2024 DSPy pipeline in this repo, run on
2026-09-24. **The 2024 project never recorded a score, so every number here is a 2026 measurement.**
Arm A re-runs the 2024 compiled program with the 2024 model. The pipeline runs on 2026 library
versions; `reeval_2026/CHANGES.md` lists every change.

`results.json` holds per-arm MAP@25, n, 95% bootstrap CI, recall@1/@25, errors, model ids, prices,
timestamps, spend and paired differences. Headline numbers and caveats are below.

## What was evaluated

The pipeline as it stood in 2024 (`src/dspy_program.py`, `IdentifyMisconceptions.forward`; the
classification step was already commented out then):

1. An LM, prompted by a DSPy ChainOfThought with the 2024-optimized instructions and 8 bootstrapped
   demos, describes the misconception behind a (question, correct answer, wrong answer) triple.
2. That description is the `near_text` query against an embedded Weaviate index of the 2,587
   misconception names, embedded with `nomic-embed-text` via Ollama.
3. The top 25 misconception ids are the prediction. The metric is MAP@25 (Kaggle definition; one
   gold id per pair, so it equals the mean reciprocal rank within the top 25).

Arms:

| arm | LM | program |
|---|---|---|
| retrieval | none: the question, correct answer and wrong answer text is the query | none |
| A | `gpt-4o-mini` (served as `gpt-4o-mini-2024-07-18`, the 2024 model) | 2024 compiled (`out/mmim_identify_misconceptions.json`) |
| B | `deepseek-ai/DeepSeek-V4-Flash` on DeepInfra | 2024 compiled, unchanged |
| C | `deepseek-ai/DeepSeek-V4-Flash` on DeepInfra | re-optimized 2026 with the 2024 optimizer settings (`arm_c_program.json`) |

Why B's model: on 2026-09-24 it listed at $0.09 / $0.18 per 1M input/output tokens on DeepInfra
against gpt-4o-mini's $0.15 / $0.60. On Artificial Analysis's leaderboard pulled the same day
(`provenance/pareto_all_models.json`), gpt-4o-mini scores 6.7 on the Intelligence Index, GPQA 0.43
and HLE 0.04. The DeepSeek V4 Flash rows score 34.8–39.5, GPQA 0.91 and HLE 0.34–0.39. Caveat:
those DeepSeek rows are its reasoning (max-effort) variants; the DeepInfra endpoint as called here
answered without a reasoning trace, and Artificial Analysis lists no separate non-reasoning row.
The per-call cost measured in this run (below) confirms the price side.

## Held-out set

- The 2024 optimizer (`BootstrapFewShotWithRandomSearch`, no valset, so validation = trainset)
  saw exactly `random.Random(9592).sample(examples, k=200)` over the per-(question, wrong answer)
  examples built from `train.csv` in file order. `reeval_2026/split.py` re-derives that sample.
  It checks the result against the compiled program: all 56 demos stored in it (48 labeled, 8
  bootstrapped) are inside the re-derived 200 (`heldout.json` → `recovery_check`). The re-derived
  sample is in `trainset_2024_recovered.json`.
- The held-out pool is every labeled (question, wrong answer) pair whose question does not appear
  anywhere in those 200 examples: 3,913 pairs from 1,674 questions. 195 questions are excluded,
  which makes the split disjoint at question level, stricter than pair level.
- `heldout.json` holds that pool as a seeded permutation (seed 20260924), with question id,
  wrong-answer letter and gold misconception id per row. All arms were scored on the full pool
  (n = 3,913), because the budget covered it.

## Results

RESULTS_TABLE

## Reproduce

```bash
# environment (Python 3.11, uv); exact versions in reeval_2026/requirements.lock.txt
uv venv -p 3.11 .venv && uv pip install --python .venv/bin/python -r reeval_2026/requirements.lock.txt
ollama pull nomic-embed-text            # Ollama server running on :11434
export EEDI_WEAVIATE_DATA_PATH=$PWD/scratch/weaviate-data

# build the Weaviate index with the unchanged 2024 ingest code
.venv/bin/python -c "from src.my_weaviate import *; c=get_weaviate_client(); define_misconception_collection(c); embed_misconceptions(c); c.close()"

# held-out set (re-derives the 2024 sample and verifies it against the compiled demos)
.venv/bin/python -m reeval_2026.split --n 3913

# arms (one at a time: the embedded Weaviate binds fixed ports). Runs are resumable.
.venv/bin/python -m reeval_2026.run_eval --arm retrieval --threads 8 --budget 10
.venv/bin/python -m reeval_2026.run_eval --arm A --threads 8 --budget 10   # needs OPENAI key file
.venv/bin/python -m reeval_2026.run_eval --arm B --threads 8 --budget 10   # needs DeepInfra key file
.venv/bin/python -m reeval_2026.reoptimize --budget 10 --threads 8
.venv/bin/python -m reeval_2026.run_eval --arm C --program out/reeval-2026-09/arm_c_program.json --threads 8 --budget 10
.venv/bin/python -m reeval_2026.analyze   # -> results.json
```

Keys are read from `~/keys/OPENAI_API_KEY_EEDI_MMIM.txt` and
`~/keys/DEEPINFRA_API_KEY_EEDI_MMIM.txt` (see `MODELS` in `reeval_2026/run_eval.py`).
`--budget` is a hard stop on the cumulative spend ledger (`spend_ledger.jsonl`).

Files: `predictions/<arm>.jsonl` holds one line per held-out row, with the 25 predicted ids, the gold
id, its rank, per-call token usage, finish reasons, the served model id, the LM's misconception
description and the row's cost. `predictions/*-pilot.jsonl` are the 20-row cost pilots. Their spend
is in the ledger but they are not part of the results.

Expected run-to-run variation: calls use temperature 0, but hosted LM outputs are not bit-stable,
so a re-run should land within the CI rather than reproduce every rank exactly.
