# Changes made to revive the October 2024 pipeline (September 2026 re-evaluation)

The pipeline code (`src/`) and the compiled program (`out/mmim_identify_misconceptions.json`,
written 2024-10-17 03:13 local time) are kept as built. The list below is every change needed to
run them on 2026 library versions, plus the evaluation-only additions.

## Environment

- Python 3.11 venv at `.venv` (uv), current releases of the pyproject's direct dependencies.
  Exact versions: `reeval_2026/requirements.lock.txt`. Key ones: dspy 3.3.1 (2024: dspy-ai 2.5.9),
  weaviate-client 4.23.1 (2024: 4.8.x), litellm 1.101.0, ollama (python) 0.6.2.
- Ollama server 0.34.3 with `nomic-embed-text:latest` (digest 0a109f422b47, pulled in 2024; same
  weights as the 2024 run).
- Embedded Weaviate binary 1.30.5 (the client's current default; 2024 used 1.26.x).

## Code changes in `src/`

1. `src/dspy_program.py`: the import `from dspy.retrieve.weaviate_rm import WeaviateRM` is
   commented out. dspy 3 removed the `dspy.retrieve` package; the import was unused.
2. `src/my_weaviate.py`, `get_weaviate_client()`:
   - adds `CLUSTER_ADVERTISE_ADDR=127.0.0.1` to the embedded server's env. Without it Weaviate
     1.30 bootstraps raft on the machine's LAN address, fails to join its own cluster and never
     reports ready (reproduced by running the binary directly: ready in 5 s with the variable,
     never ready without it).
   - honours an optional `EEDI_WEAVIATE_DATA_PATH` for the persistence directory. Unset, the 2024
     default (`~/.local/share/weaviate`, shared with other projects on this machine) applies. The
     re-evaluation sets it to `scratch/weaviate-data` and rebuilds the index from
     `misconception_mapping.csv` with the unchanged `define_misconception_collection` +
     `embed_misconceptions` (2,587 objects), instead of reusing the 2024 on-disk index.

## Loading the 2024 compiled program under dspy 3 (`reeval_2026/run_eval.py`)

The saved JSON cannot be loaded by dspy 3's `Module.load` directly:

- it has no `metadata` block (dspy 3 `load` requires one), so `load_state` is called directly;
- dspy 2.5 saved each `ChainOfThought` under its attribute name with the reasoning-augmented
  signature in `extended_signature`; dspy 3 nests the predictor as `<name>.predict` and raises
  `NotImplementedError` on `extended_signature`. `translate_2024_state` renames the key and uses the
  saved `extended_signature` (the signature 2.5 actually prompted with) as `signature`.
  Instructions, field prefixes/descriptions, and all demos are carried over unchanged.

What this means for the prompt: the instructions, field descriptions and the 8 bootstrapped
(reasoning-augmented) demos are identical to 2024. The other 24 labeled demos in the
`generate_misconception_query` predictor carry only a serialized `question` and no output field;
both dspy 2.5.9's and dspy 3's chat adapters drop demos that have no output field (2.5.9:
`dspy/adapters/chat_adapter.py` `ChatAdapter.format`, checked against the 2.5.9 wheel from PyPI;
3.3.1: `Adapter.format_demos`), so they were not in the 2024 prompt and are not in the 2026 prompt. The chat adapter's message template itself is
dspy 3's, not 2.5.9's, so the wording around the fields differs from 2024. This is the one known
difference between "the 2024 system" and what Arm A runs.

## LM settings

- `temperature=0.0, max_tokens=1000` are set explicitly: they were dspy 2.5.9's `dspy.LM`
  defaults (`dspy/clients/lm.py`, checked in the 2.5.9 wheel) (what the 2024 code got implicitly); dspy 3 defaults both to `None`. Every call's
  `finish_reason` is logged; see `results.json` for whether any call hit the token limit.
- `cache=False` so every evaluation row is a real, billed API call; `num_retries=5`.
- Arm A: `openai/gpt-4o-mini`; the API served snapshot `gpt-4o-mini-2024-07-18` (logged per call).
- Credentials: per-consumer keys minted for this run (`~/keys/OPENAI_API_KEY_EEDI_MMIM.txt`, OpenAI
  project `eedi-mmim` with an $8 hard monthly cap; `~/keys/DEEPINFRA_API_KEY_EEDI_MMIM.txt`,
  DeepInfra token `eedi-mmim`, no provider-side limit available, contained by the spend ledger's
  hard stop).

## Arm C (re-optimization)

`reeval_2026/reoptimize.py` runs `BootstrapFewShotWithRandomSearch` with the 2024 settings
(max_labeled_demos=32, max_bootstrapped_demos=8, num_candidate_programs=24, metric_threshold=1,
max_rounds=1, metric = `score_identified_misconceptions`) on the recovered 2024 trainset, with the
Arm B model. Only `num_threads` differs (2024: 128), which affects wall-clock, not the search.

## Not changed

The classification step in `IdentifyMisconceptions.forward` was already commented out in 2024;
the evaluated pipeline is: LM writes a misconception description (ChainOfThought) → the
description is the Weaviate `near_text` query → the top 25 misconception ids are the prediction.
