"""Evaluate the 2024 compiled DSPy program on the held-out set (2026 re-eval).

Arms:
  retrieval  - no LM: the question/answer text itself is the Weaviate query (embedding rank only)
  A          - the 2024 compiled program, LM = gpt-4o-mini (the 2024 model)
  B          - the same compiled program and retrieval, LM swapped to DeepSeek V4 Flash (DeepInfra)
  C          - a program re-optimized with the Arm B model (see reoptimize.py), same eval path

Per-example predictions are appended to out/reeval-2026-09/predictions/<arm>.jsonl, and a run is
resumable: rows already present are skipped. Every paid call's cost is appended to
out/reeval-2026-09/spend_ledger.jsonl and the run stops before the ledger total passes --budget.

Usage: .venv/bin/python -m reeval_2026.run_eval --arm A --limit 20 --threads 16 --budget 10
"""
import argparse
import datetime as dt
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dspy
from loguru import logger

from src.constants import PROJECT_ROOT, mmim_data_path
from src.data_loading import load_train_data
from src.models import AnswerDesignation

OUT = PROJECT_ROOT / "out" / "reeval-2026-09"
KEYS = Path.home() / "keys"
COMPILED_2024 = PROJECT_ROOT / "out" / "mmim_identify_misconceptions.json"

# USD per 1M tokens, read 2026-09-24 (provenance in README). DeepInfra also returns
# usage.estimated_cost per call; that value is preferred when present.
MODELS = {
    "A": {"model": "openai/gpt-4o-mini", "key": "OPENAI_API_KEY_EEDI_MMIM.txt",
          "price_in": 0.15, "price_cached_in": 0.075, "price_out": 0.60},
    "B": {"model": "deepinfra/deepseek-ai/DeepSeek-V4-Flash", "key": "DEEPINFRA_API_KEY_EEDI_MMIM.txt",
          "price_in": 0.09, "price_cached_in": 0.018, "price_out": 0.18},
}
MODELS["C"] = MODELS["B"]

LEDGER = OUT / "spend_ledger.jsonl"
_ledger_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())


def make_lm(arm: str) -> dspy.LM:
    spec = MODELS[arm]
    # temperature=0.0 / max_tokens=1000 are dspy 2.5.9's dspy.LM defaults, i.e. what the 2024
    # run used implicitly; dspy 3 defaults both to None, so they are pinned here.
    return dspy.LM(spec["model"], api_key=(KEYS / spec["key"]).read_text().strip(),
                   temperature=0.0, max_tokens=1000, cache=False, num_retries=5)


def call_cost(arm: str, usage: dict) -> float:
    if usage.get("estimated_cost") is not None:
        return float(usage["estimated_cost"])
    spec = MODELS[arm]
    prompt = usage.get("prompt_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    completion = usage.get("completion_tokens") or 0
    return ((prompt - cached) * spec["price_in"] + cached * spec["price_cached_in"]
            + completion * spec["price_out"]) / 1e6


def translate_2024_state(state: dict) -> dict:
    """Map a dspy 2.5 saved state onto dspy 3 parameter names.

    dspy 2.5 ChainOfThought was a Predict subclass saved under the module attribute name, with the
    reasoning-extended signature in `extended_signature`. dspy 3 nests it as `<name>.predict` and
    refuses `extended_signature`, so the extended signature (the one 2.5 actually prompted with)
    becomes `signature`. Demos, instructions and field prefixes are carried over unchanged.
    """
    out = {}
    for name, sub in state.items():
        if "extended_signature" in sub:
            sub = dict(sub)
            sub["signature"] = sub.pop("extended_signature")
            out[f"{name}.predict"] = sub
        else:
            out[name] = sub
    return out


def load_program(path: Path):
    from src.dspy_program import IdentifyMisconceptions
    program = IdentifyMisconceptions(num_misconceptions_to_consider=25)
    state = json.loads(path.read_text())
    # load_state deep-copies the module for a trial run; detach the live Weaviate client meanwhile
    # so the copy does not clone (and later try to tear down) the embedded server handle.
    client = program.retrieve_misconceptions.weaviate_client
    program.retrieve_misconceptions.weaviate_client = None
    if "metadata" in state:  # saved by dspy 3 (Arm C)
        program.load(str(path))
    else:
        program.load_state(translate_2024_state(state))
    program.retrieve_misconceptions.weaviate_client = client
    return program


def plain(obj):
    """litellm usage objects -> plain dicts (recursively)."""
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return plain(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return plain(vars(obj))
    return obj


def heldout_examples(limit: int | None):
    rows = json.loads((OUT / "heldout.json").read_text())["rows"]
    questions = {q.question_id: q for q in load_train_data(str(mmim_data_path / "train.csv"))}
    rows = rows[:limit] if limit else rows
    return [(r, questions[r["question_id"]]) for r in rows]


def rank_of(ids, truth):
    for i, mid in enumerate(ids[:25], start=1):
        if mid == truth:
            return i
    return None


def run_retrieval(row, question, client):
    from src.my_weaviate import retrieve_misconceptions
    designation = AnswerDesignation(row["wrong_answer"])
    query = (f"Question: {question.question_text}\n"
             f"Correct answer: {question.correct_answer_text}\n"
             f"Wrong answer: {question.answer_text_for_designation(designation)}")
    ids = [m.misconception_id for m in retrieve_misconceptions(client=client, query=query, k=25)]
    return {"predicted_ids": ids, "usage": [], "usd": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["retrieval", "A", "B", "C"])
    parser.add_argument("--program", default=str(COMPILED_2024))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--budget", type=float, required=True, help="stop before the ledger total passes this (USD)")
    parser.add_argument("--tag", default="", help="suffix for the predictions file (e.g. pilot)")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="WARNING")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    pred_path = OUT / "predictions" / f"{args.arm}{('-' + args.tag) if args.tag else ''}.jsonl"
    done = set()
    if pred_path.exists():
        done = {(r["question_id"], r["wrong_answer"]) for r in map(json.loads, pred_path.read_text().splitlines())}

    todo = [(r, q) for r, q in heldout_examples(args.limit) if (r["question_id"], r["wrong_answer"]) not in done]
    print(f"arm {args.arm}: {len(done)} done, {len(todo)} to run, ledger ${ledger_total():.4f}")

    program = lm = None
    if args.arm == "retrieval":
        from src.my_weaviate import get_weaviate_client
        client = get_weaviate_client()
    else:
        # the program's own RetrieveMisconceptions opens the embedded Weaviate client (as built)
        program = load_program(Path(args.program))
        client = program.retrieve_misconceptions.weaviate_client
        lm = make_lm(args.arm)

    write_lock = threading.Lock()
    spent = {"usd": ledger_total()}

    def one(row, question):
        if spent["usd"] >= args.budget:
            raise BudgetExceeded(f"ledger ${spent['usd']:.4f} >= budget ${args.budget}")
        record = {"question_id": row["question_id"], "wrong_answer": row["wrong_answer"],
                  "misconception_id": row["misconception_id"], "arm": args.arm,
                  "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
        if args.arm == "retrieval":
            record.update(run_retrieval(row, question, client))
            record["usage"] = []
        else:
            designation = AnswerDesignation(row["wrong_answer"])
            local_lm = lm.copy()  # own history per call, so usage attribution is exact under threads
            with dspy.context(lm=local_lm):
                try:
                    pred = program(question=question, wrong_answer_designation=designation)
                    record["predicted_ids"] = pred.misconception_ids
                    record["error"] = None
                except Exception as exc:  # recorded explicitly, scored as a miss, counted in results
                    record["predicted_ids"] = []
                    record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            usages = [plain(h.get("usage")) or {} for h in local_lm.history]
            record["usage"] = [{k: u.get(k) for k in ("prompt_tokens", "completion_tokens")} |
                               {"cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
                               for u in usages]
            record["response_model"] = [h.get("response_model") for h in local_lm.history]
            record["finish_reason"] = [
                getattr(h["response"].choices[0], "finish_reason", None) if h.get("response") is not None else None
                for h in local_lm.history]
            record["misconception_description"] = (
                local_lm.history[-1]["outputs"][0] if local_lm.history else None)
            usd = sum(call_cost(args.arm, u) for u in usages)
            record["usd"] = usd
            with _ledger_lock:
                with LEDGER.open("a") as f:
                    f.write(json.dumps({"arm": args.arm, "tag": args.tag, "usd": usd,
                                        "question_id": row["question_id"], "wrong_answer": row["wrong_answer"]}) + "\n")
                spent["usd"] += usd
        record["rank"] = rank_of(record["predicted_ids"], row["misconception_id"])
        record["ap25"] = 1.0 / record["rank"] if record["rank"] else 0.0
        with write_lock:
            with pred_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    n = 0
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(one, r, q) for r, q in todo]
            for fut in as_completed(futures):
                rec = fut.result()
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)} ledger ${spent['usd']:.4f}")
    finally:
        client.close()
    print(f"arm {args.arm}: wrote {n}, ledger ${ledger_total():.4f}")


if __name__ == "__main__":
    main()
