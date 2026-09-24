---
date: 2026-09-25T00:41:38
title: Held-out Re-evaluation of 2024 Pipeline
type: progress
author: "Claude"
ai_generated: true
ai_model: "claude-opus-5-5"
session_id: c683d108-92c4-4599-b9eb-f33343793dd6
---

## Summary

The October 2024 DSPy misconception pipeline now has its first recorded score. On 3,913 held-out
train.csv pairs it reaches MAP@25 0.158 [0.148, 0.167] with gpt-4o-mini, 0.181 [0.171, 0.191] with
DeepSeek V4 Flash, and 0.195 [0.185, 0.205] when re-optimized for V4 Flash. A no-LM retrieval
baseline scores 0.105 [0.097, 0.112]. Everything was measured on 2026-09-24/25 and totalled $2.73
of third-party spend.

## What Was Done

Dispatched by bid-pipeline-lead (session 44c23848) to produce reproducible numbers for a
technical write-up in the BfS RODOS tender (3626S62540). Hard requirements: honesty, a $10 cap,
and no number presented as a 2024 result.

- **Environment revived** on dspy 3.3.1 / weaviate-client 4.23.1 / Weaviate 1.30.5 embedded /
  Ollama nomic-embed-text, with minimal code changes (`reeval_2026/CHANGES.md`):
  - a dead `dspy.retrieve` import;
  - `CLUSTER_ADVERTISE_ADDR=127.0.0.1`, without which embedded Weaviate 1.30 bootstraps raft on
    the LAN IP and never becomes ready;
  - an optional project-local data path.
  The 2024 compiled JSON needed a state translation: dspy 2.5 `extended_signature` became dspy 3
  `<name>.predict` + `signature`. I checked against the dspy 2.5.9 wheel that output-less labeled
  demos were dropped in 2024 as well. The 2.5.9 LM defaults (temperature 0, max_tokens 1000) are
  now pinned explicitly.
- **Split recovered, not guessed.** `random.Random(9592).sample(examples, 200)` re-derived the
  optimizer's train/val set. All 56 demos stored in the compiled program fall inside it. The
  held-out pool is every labeled pair from questions absent from those 200: 3,913 pairs from
  1,674 questions.
- **Arms**: retrieval-only, A (2024 program + gpt-4o-mini, served as the 2024-07-18 snapshot),
  B (same program + DeepSeek V4 Flash on DeepInfra), C (the 2024 optimizer settings re-run with
  V4 Flash). All were scored on the full pool, and paired bootstraps are in `results.json`.
- **Keys**: minted a per-consumer OpenAI project `eedi-mmim` with an $8 cap, and a DeepInfra
  token `eedi-mmim`, which has no provider-side limit. The spend ledger has a hard stop. The
  ledger and OpenAI's billed cost agree ($1.44 vs $1.45).
- A 20-row pilot projected about $1.7 for A+B on the full pool, which is why every arm ran all
  rows rather than a sample.

## Steering and reasoning

- The allocator session asked for a quiet machine during Christo's call (16:54–17:35). Arm B
  was stopped. It was resumable, so nothing was lost. It resumed at 17:41 with 8 threads and
  heavy jobs were gated on load < 500.
- Arm C's optimizer took about 4.5 h under machine load, far over my 45-minute estimate. It
  finished, but dspy 3.3.1's `save()` crashed on `Retrieve.dump_state(json_mode=...)`. The log
  and the dspy source show the selected candidate was seed −3, the zero-shot `reset_copy()`, so
  the program was rebuilt exactly instead of re-run. That gives the substantive finding: the
  2024 demos, bootstrapped with gpt-4o-mini, slightly hurt V4 Flash (C − B = +0.014
  [+0.006, +0.023]).
- Transient Ollama "socket is not connected" errors under concurrency were first silently
  dropping retrieval rows. I then noticed they would be scored as model misses in the LM arms
  and fixed it before arm A ran. Retrieval now retries with backoff, and a row that still fails
  stays un-recorded, so a resumed run re-does it rather than scoring it as a miss.

## Implications

`out/reeval-2026-09/README.md` + `results.json` are the dispatcher's source for the tender prose.
Caveats it must carry are listed in the README:
- Arm A uses dspy 3's chat template, not 2.5.9's.
- Errors are scored as misses (B 3, C 28), caused by the kept 2024 max_tokens=1000, so B and C
  are slightly understated.
- Arm C's program is a rebuild of the selected candidate.
- The Artificial Analysis quality rows for V4 Flash are its reasoning variants.

## Open Threads

- The dispatcher drafts the tender prose; this session wrote none.
- A context number worth adding if the tender compares against the field: Kaggle leaderboard
  scores. I did not verify any; look them up before citing.
- If a stronger result is wanted: raise max_tokens (it removes the C errors), restore the
  commented-out classification/rerank step, or add a better retrieval query. None of these
  were in scope.
- The DeepInfra token `eedi-mmim` stays live with no provider-side limit. Revoke it when the
  write-up is done if no further runs are planned.
