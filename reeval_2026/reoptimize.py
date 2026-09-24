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

from reeval_2026.run_eval import LEDGER, OUT, BudgetExceeded, call_cost, ledger_total, make_lm, plain
from reeval_2026.split import recover_2024_trainset
from src.constants import mmim_data_path
from src.data_loading import load_train_data
from src.dspy_program import IdentifyMisconceptions, score_identified_misconceptions

PROGRAM_OUT = OUT / "arm_c_program.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

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

    client = compiled.retrieve_misconceptions.weaviate_client
    compiled.retrieve_misconceptions.weaviate_client = None
    compiled.save(str(PROGRAM_OUT))
    compiled.retrieve_misconceptions.weaviate_client = client
    client.close()
    print(json.dumps({"saved": str(PROGRAM_OUT), "optimizer_calls": state["calls"],
                      "ledger_total_usd": round(state["usd"], 4)}))


if __name__ == "__main__":
    main()
