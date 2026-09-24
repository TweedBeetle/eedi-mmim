"""Aggregate per-row predictions into out/reeval-2026-09/results.json (2026 re-eval).

MAP@25 with one gold misconception per (question, wrong answer) is the mean over rows of 1/rank of
the gold id within the top 25 predictions (0 if absent), matching src/dspy_program.py's metric and
the Kaggle definition. CIs are percentile bootstraps over rows (10,000 resamples, fixed seed);
arm differences use a paired bootstrap over the rows both arms scored.

Usage: .venv/bin/python -m reeval_2026.analyze
"""
import datetime as dt
import json

import numpy as np

from reeval_2026.run_eval import LEDGER, MODELS, OUT

RESAMPLES = 10_000
SEED = 20260924
ARMS = {
    "retrieval": {"description": "retrieval only: question + correct + wrong answer text embedded with nomic-embed-text, "
                                 "nearest 25 misconceptions; no LM", "lm": None, "program": None},
    "A": {"description": "2024 compiled DSPy program as built, LM gpt-4o-mini", "program": "out/mmim_identify_misconceptions.json"},
    "B": {"description": "same 2024 compiled program and retrieval, LM DeepSeek V4 Flash (DeepInfra)",
          "program": "out/mmim_identify_misconceptions.json"},
    "C": {"description": "program re-optimized in 2026 with the 2024 optimizer settings, LM DeepSeek V4 Flash",
          "program": "out/reeval-2026-09/arm_c_program.json"},
}


def load(arm):
    path = OUT / "predictions" / f"{arm}.jsonl"
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[(r["question_id"], r["wrong_answer"])] = r
    return rows


def ci(values, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows, rng):
    ap = [r["ap25"] for r in rows.values()]
    ranks = [r["rank"] for r in rows.values()]
    usd = sum(r.get("usd", 0.0) for r in rows.values())
    models = sorted({m for r in rows.values() for m in (r.get("response_model") or []) if m})
    finish = sorted({f for r in rows.values() for f in (r.get("finish_reason") or []) if f})
    return {
        "n": len(rows),
        "map_at_25": float(np.mean(ap)),
        "map_at_25_ci95": ci(ap, rng),
        "recall_at_1": float(np.mean([rk == 1 for rk in ranks])),
        "recall_at_25": float(np.mean([rk is not None for rk in ranks])),
        "errors": sum(1 for r in rows.values() if r.get("error")),
        "cost_usd_eval": round(usd, 4),
        "response_models": models,
        "finish_reasons": finish,
        "first_row_ts": min(r["ts"] for r in rows.values()),
        "last_row_ts": max(r["ts"] for r in rows.values()),
    }


def paired(a, b, rng):
    keys = sorted(set(a) & set(b))
    d = np.array([b[k]["ap25"] - a[k]["ap25"] for k in keys])
    return {"n": len(keys), "delta_map_at_25": float(d.mean()), "delta_ci95": ci(d, rng)}


def main():
    rng = np.random.default_rng(SEED)
    data = {arm: load(arm) for arm in ARMS}
    heldout = json.loads((OUT / "heldout.json").read_text())
    lm_keys = set(data["A"]) if data["A"] else set()

    ledger = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    spend = {}
    for e in ledger:
        k = f'{e["arm"]}{("-" + e["tag"]) if e.get("tag") else ""}'
        spend[k] = spend.get(k, 0.0) + e["usd"]

    arms = {}
    for arm, rows in data.items():
        if rows is None:
            continue
        entry = dict(ARMS[arm])
        if arm in MODELS:
            entry["lm"] = MODELS[arm]["model"]
            entry["price_usd_per_mtok"] = {k: MODELS[arm][k] for k in ("price_in", "price_cached_in", "price_out")}
        entry["all_rows"] = summarize(rows, rng)
        if arm == "retrieval" and lm_keys:
            entry["same_rows_as_lm_arms"] = summarize({k: rows[k] for k in lm_keys if k in rows}, rng)
        arms[arm] = entry

    comparisons = {}
    for a, b in [("retrieval", "A"), ("retrieval", "B"), ("A", "B"), ("B", "C"), ("A", "C")]:
        if data.get(a) and data.get(b):
            comparisons[f"{b}_minus_{a}"] = paired(data[a], data[b], rng)

    results = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "All numbers produced in September 2026 by re-running the October 2024 pipeline; "
                "none of them is a 2024 result. No evaluation score was recorded in 2024.",
        "metric": "MAP@25 (one gold misconception per question/wrong-answer pair = mean reciprocal rank "
                  "within top 25); 95% percentile bootstrap CI, 10,000 resamples",
        "heldout": {"source": "train.csv labeled pairs from questions the 2024 optimizer never saw",
                    "pool_size": heldout["pool_size"], "seed": heldout["seed"],
                    "excluded_questions": heldout["excluded_questions_seen_by_2024_optimizer"],
                    "recovery_check": heldout["recovery_check"], "row_ids": "out/reeval-2026-09/heldout.json"},
        "arms": arms,
        "paired_differences": comparisons,
        "spend_usd": {"by_run": {k: round(v, 4) for k, v in spend.items()},
                      "total": round(sum(spend.values()), 4), "cap": 10.0},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
