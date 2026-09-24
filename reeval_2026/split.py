"""Recover the 2024 optimizer's training sample and build a held-out evaluation set (2026 re-eval).

The 2024 run (src/dspy_program.py:main) built one dspy.Example per (question, wrong answer) over
train.csv in file order, then took random.Random(9592).sample(examples, k=200) and passed those 200
as `trainset` to BootstrapFewShotWithRandomSearch with no `valset`, so DSPy used the same 200 for
candidate scoring. Everything the optimizer saw is therefore inside those 200 examples.

This script re-derives the 200 with the same code path, checks the recovery against the demos
stored in the compiled program, and then draws the held-out set from questions that do not occur
anywhere in the 200 (question-level disjointness, stricter than pair-level), keeping only pairs
that carry a labeled misconception.

Usage: .venv/bin/python -m reeval_2026.split [--n N] [--seed S]
"""
import argparse
import json
import random

from src.constants import PROJECT_ROOT, mmim_data_path
from src.data_loading import load_train_data

OUT = PROJECT_ROOT / "out" / "reeval-2026-09"
COMPILED = PROJECT_ROOT / "out" / "mmim_identify_misconceptions.json"


def all_examples(questions):
    examples = []
    for question in questions:
        examples.extend(question.as_dspy_examples())
    return examples


def recover_2024_trainset(questions):
    return random.Random(9592).sample(all_examples(questions), k=200)


def key(ex):
    return (ex.question.question_id, ex.wrong_answer_designation.value)


def verify_against_compiled(trainset, questions):
    """Every demo in the compiled program must come from the recovered 200."""
    train_keys = {key(e) for e in trainset}
    by_text = {}
    for e in trainset:
        q = e.question
        by_text.setdefault((q.question_text, q.answer_text_for_designation(e.wrong_answer_designation)), set()).add(key(e))

    compiled = json.loads(COMPILED.read_text())
    report = {"labeled_demos": 0, "labeled_in_trainset": 0, "augmented_demos": 0, "augmented_in_trainset": 0}
    for predictor in ("generate_misconception_query", "classify_misconception"):
        for demo in compiled[predictor]["demos"]:
            if demo.get("augmented"):
                report["augmented_demos"] += 1
                if (demo["question"], demo["wrong_answer"]) in by_text:
                    report["augmented_in_trainset"] += 1
            else:
                report["labeled_demos"] += 1
                q = json.loads(demo["question"])
                if (q["question_id"], demo["wrong_answer_designation"]) in train_keys:
                    report["labeled_in_trainset"] += 1
    report["recovery_verified"] = (
        report["labeled_demos"] == report["labeled_in_trainset"]
        and report["augmented_demos"] == report["augmented_in_trainset"]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="held-out pairs to draw")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    questions = load_train_data(str(mmim_data_path / "train.csv"))
    trainset = recover_2024_trainset(questions)
    report = verify_against_compiled(trainset, questions)
    if not report["recovery_verified"]:
        raise RuntimeError(f"2024 trainset recovery does not match compiled demos: {report}")

    seen_questions = {e.question.question_id for e in trainset}
    pool = [
        e for e in all_examples(questions)
        if e.question.question_id not in seen_questions
        and e.question.misconception_id_for_answer_designation(e.wrong_answer_designation) is not None
    ]
    if args.n > len(pool):
        raise ValueError(f"requested {args.n} > pool {len(pool)}")
    heldout = random.Random(args.seed).sample(pool, k=args.n)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trainset_2024_recovered.json").write_text(json.dumps(
        [{"question_id": k[0], "wrong_answer": k[1]} for k in map(key, trainset)], indent=1))
    (OUT / "heldout.json").write_text(json.dumps({
        "seed": args.seed,
        "n": args.n,
        "pool_size": len(pool),
        "train_csv_questions": len(questions),
        "labeled_pairs_total": sum(
            1 for e in all_examples(questions)
            if e.question.misconception_id_for_answer_designation(e.wrong_answer_designation) is not None),
        "excluded_questions_seen_by_2024_optimizer": len(seen_questions),
        "recovery_check": report,
        "rows": [
            {"question_id": e.question.question_id, "wrong_answer": e.wrong_answer_designation.value,
             "misconception_id": e.question.misconception_id_for_answer_designation(e.wrong_answer_designation)}
            for e in heldout
        ],
    }, indent=1))
    print(json.dumps({k: v for k, v in json.loads((OUT / "heldout.json").read_text()).items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
