"""Arm C: re-run the 2024 optimizer with the Arm B model (2026 re-eval).

Same pipeline class, same metric, same trainset (the recovered random.Random(9592) sample of 200),
same BootstrapFewShotWithRandomSearch settings as src/dspy_program.py:main, except num_threads
(128 -> --threads; thread count does not change what is optimized, only wall-clock) and the LM.
The result is saved as a dspy 3 state file and evaluated by run_eval.py --arm C --program <it>.

Spend: every LM call is priced from litellm's usage (DeepInfra's estimated_cost) and appended to
the shared spend ledger; the metric raises once the ledger passes --budget, which aborts the run.

Usage: .venv/bin/python -m reeval_2026.reoptimize --budget 9.5 --threads 32
"""
import argparse
import json
import threading

import dspy
import litellm
from loguru import logger

from reeval_2026.run_eval import (LEDGER, OUT, BudgetExceeded, call_cost, install_retrieval_retry, ledger_total,
                                  make_lm, plain)
from reeval_2026.split import recover_2024_trainset
from src.constants import mmim_data_path
from src.data_loading import load_train_data
from src.dspy_program import IdentifyMisconceptions, score_identified_misconceptions

PROGRAM_OUT = OUT / "arm_c_program.json"
SCORES_OUT = OUT / "arm_c_optimizer_scores.json"


def patch_retrieve_dump_state():
    """dspy 3.3.1's Module.save calls dump_state(json_mode=...) on every parameter, but its own
    dspy.Retrieve.dump_state takes no arguments, so saving any program containing a Retrieve raises
    TypeError (this lost the first Arm C run's save). Accept and ignore the keyword."""
    original = dspy.Retrieve.dump_state
    if getattr(original, "_patched", False):
        return

    def dump_state(self, *args, **kwargs):
        return original(self)
    dump_state._patched = True
    dspy.Retrieve.dump_state = dump_state


def save_program(program, path):
    patch_retrieve_dump_state()
    client = program.retrieve_misconceptions.weaviate_client
    program.retrieve_misconceptions.weaviate_client = None
    program.save(str(path))
    program.retrieve_misconceptions.weaviate_client = client


def scores_from_log(log_path):
    """Candidate scores in seed order (-3, -2, -1, 0..23) and the winning seed, from the optimizer's stdout."""
    import re
    text = open(log_path, errors="replace").read()
    last = re.findall(r"Scores so far: \[([^\]]*)\]", text)[-1]
    scores = [float(x) for x in last.split(",")]
    best = re.findall(r"New best score: ([0-9.]+) for seed (-?[0-9]+)", text)
    return {"seeds": list(range(-3, -3 + len(scores))), "scores_pct_map25_on_trainset": scores,
            "new_best_events": [{"score": float(a), "seed": int(b)} for a, b in best]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--save-selected-from-log", default=None, metavar="LOG",
                        help="recovery: read a finished run's stdout log; if the selected candidate was seed -3 "
                             "(zero-shot = student.reset_copy(), no demos), rebuild and save it without re-running")
    args = parser.parse_args()

    if args.save_selected_from_log:
        info = scores_from_log(args.save_selected_from_log)
        winner = info["new_best_events"][-1]["seed"]
        if winner != -3:
            raise RuntimeError(f"selected seed {winner} is not the zero-shot candidate; it cannot be rebuilt, re-run")
        info["selected_seed"] = winner
        info["selected_program"] = "seed -3 = student.reset_copy(): uncompiled IdentifyMisconceptions, no demos"
        SCORES_OUT.write_text(json.dumps(info, indent=1))
        program = IdentifyMisconceptions(num_misconceptions_to_consider=25).reset_copy()
        save_program(program, PROGRAM_OUT)
        program.retrieve_misconceptions.weaviate_client.close()
        print(json.dumps({"saved": str(PROGRAM_OUT), **info}))
        return

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="WARNING")

    lock = threading.Lock()
    state = {"usd": ledger_total(), "calls": 0}

    def on_success(kwargs, response, start_time, end_time):
        usd = call_cost("C", plain(response.usage))
        with lock:
            state["usd"] += usd
            state["calls"] += 1
            with LEDGER.open("a") as f:
                f.write(json.dumps({"arm": "C", "tag": "optimize", "usd": usd}) + "\n")

    litellm.success_callback = [on_success]

    def metric(example, pred, trace=None):
        if state["usd"] >= args.budget:
            raise BudgetExceeded(f"ledger ${state['usd']:.4f} >= ${args.budget}")
        return score_identified_misconceptions(example, pred, trace)

    install_retrieval_retry()
    questions = load_train_data(str(mmim_data_path / "train.csv"))
    trainset = recover_2024_trainset(questions)

    dspy.settings.configure(lm=make_lm("C"))
    program = IdentifyMisconceptions(num_misconceptions_to_consider=25)
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=metric,
        num_threads=args.threads,
        max_labeled_demos=32,
        max_bootstrapped_demos=8,
        num_candidate_programs=24,
        metric_threshold=1.,
        max_rounds=1,
    )
    compiled = optimizer.compile(program, trainset=trainset)

    save_program(compiled, PROGRAM_OUT)
    compiled.retrieve_misconceptions.weaviate_client.close()
    print(json.dumps({"saved": str(PROGRAM_OUT), "optimizer_calls": state["calls"],
                      "ledger_total_usd": round(state["usd"], 4)}))


if __name__ == "__main__":
    main()
