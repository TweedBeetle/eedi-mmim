import argparse
import concurrent
import json
import random
from functools import partial
from typing import List, Dict, Optional, Set

import dspy
from dspy import Module, Signature, InputField, OutputField, ChainOfThought, Predict, Example
from dspy.datasets import Dataset
from dspy.retrieve.weaviate_rm import WeaviateRM
from dspy.teleprompt import BootstrapFinetune
from loguru import logger
from tqdm import tqdm

from src.models import Misconception, AnswerDesignation, Question, MisconceptionsForAnswer
from src.my_weaviate import get_weaviate_client, retrieve_misconceptions
from src.data_loading import load_misconceptions, load_train_data
from src.constants import mmim_data_path, PROJECT_ROOT
import ollama


# Define Signatures

class DescribeMisconceptionSignature(Signature):
    """Identify the misconception that lead to the wrong answer."""

    # examples:
    # Does not know that angles in a triangle sum to 180 degrees
    # Uses dividing fractions method for multiplying fractions
    # Believes there are 100 degrees in a full turn
    # Confuses obtuse and acute angles
    # When reading value from graph, reads from the wrong axes.

    question = InputField(description="The original mathematical question.")
    right_answer = InputField(description="The correct answer given to the question.")
    wrong_answer = InputField(description="The incorrect answer given to the question.")
    misconception_description = OutputField(
        description="The misconception that lead to the wrong answer."
    )


class RetrieveMisconceptionsSignature(Signature):
    """Retrieve top-k misconceptions based on the misconception query."""
    misconception_query = InputField(description="The search query to retrieve misconceptions.")
    misconceptions = OutputField(description="List of retrieved Misconception objects.")


class ClassifyMisconceptionSignature(Signature):
    """Decide whether the misconception is the specific underlying reason for the wrong answer to the question."""
    question = InputField(description="The original mathematical question.")
    right_answer = InputField(description="The correct answer given to the question.")
    wrong_answer = InputField(description="The incorrect answer that was chosen.")
    misconception = InputField(
        description="The misconception to consider as the potential specific underlying reason for the wrong answer."
    )

    misconception_underlies_wrong_answer = OutputField(
        description="Boolean indicating whether the misconception is the specific underlying reason for the wrong answer."
    )


class RetrieveMisconceptions(dspy.Retrieve):
    def __init__(self, k: int = 3):
        super().__init__(k=k)

        self.weaviate_client = get_weaviate_client()

    def forward(self, query_or_queries: str) -> dspy.Prediction:
        if isinstance(query_or_queries, list):
            raise NotImplementedError("This retrieval module does not support batch queries.")

        query = query_or_queries

        misconceptions = retrieve_misconceptions(client=self.weaviate_client, query=query, k=self.k)
        return misconceptions


# Define the Main Pipeline

class IdentifyMisconceptions(Module):
    """Multi-Hop Misconception Identification and Management Pipeline."""

    def __init__(self, num_misconceptions_to_consider: int = 25):
        super().__init__()
        self.generate_misconception_query = dspy.ChainOfThought(DescribeMisconceptionSignature)

        self.retrieve_misconceptions = RetrieveMisconceptions(
            k=num_misconceptions_to_consider
        )

        self.classify_misconception = dspy.ChainOfThought(ClassifyMisconceptionSignature)

    def forward(self, question: Question, wrong_answer_designation: AnswerDesignation) -> MisconceptionsForAnswer:
        identified_misconception_ids = []

        wrong_answer_text = question.answer_text_for_designation(wrong_answer_designation)

        logger.info(f"Processing Misconceptions for Incorrect Answer Option: {wrong_answer_text}")

        # Step 1: Generate Misconception Query
        misconception_description = self.generate_misconception_query(
            question=question.question_text,
            right_answer=question.correct_answer_text,
            wrong_answer=wrong_answer_text,
        ).completions[0].misconception_description

        logger.trace(
            f"Generated Misconception description for Option {wrong_answer_text}: {misconception_description}"
        )

        # @todo:0: generate multiple misconception descriptions (eg as a numbered list)

        # Step 2: Retrieve Top-k Misconceptions
        misconceptions = self.retrieve_misconceptions(
            query_or_queries=misconception_description,

        )

        return MisconceptionsForAnswer(
            question_id=question.question_id,
            answer_designation=wrong_answer_designation,
            misconception_ids=[misconception.misconception_id for misconception in misconceptions]
        )

        # logger.debug(f"Retrieved {len(misconceptions)} Misconceptions for Option {wrong_answer_text}")
        #
        # # Step 3: Classify Each Misconception
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_to_misconception = {
        #         executor.submit(
        #             self.classify_misconception,
        #             question=question.question_text,
        #             right_answer=question.correct_answer_text,
        #             wrong_answer=wrong_answer_text,
        #             misconception=misconception.misconception_name
        #         ): misconception for misconception in misconceptions
        #     }
        #
        #     for future in concurrent.futures.as_completed(future_to_misconception):
        #         misconception = future_to_misconception[future]
        #         try:
        #             result = future.result()
        #             misconception_underlies_wrong_answer = result.completions[
        #                 0].misconception_underlies_wrong_answer
        #
        #             logger.trace(
        #                 f"Include Misconception {misconception.misconception_id}: {misconception_underlies_wrong_answer}"
        #             )
        #
        #             if misconception_underlies_wrong_answer:
        #                 identified_misconception_ids.append(
        #                     misconception.misconception_id
        #                 )
        #                 logger.trace(
        #                     f"Included Misconception ID {misconception.misconception_id} for Option {wrong_answer_text}"
        #                 )
        #         except Exception as exc:
        #             logger.error(
        #                 f"Error classifying Misconception ID {misconception.misconception_id} for Option {wrong_answer_text}: {exc}"
        #             )
        #
        # return MisconceptionsForAnswer(
        #     question_id=question.question_id,
        #     answer_designation=wrong_answer_designation,
        #     misconception_ids=identified_misconception_ids
        # )


def score_identified_misconceptions(
        example,
        pred: MisconceptionsForAnswer,
        trace=None,
        prediction_limit=25,
) -> float | bool:
    # example.question: Question
    # example.wrong_answer_designation: AnswerDesignation

    # Extract ground truth misconception IDs for the specific answer designation
    ground_truth_id = example.question.misconception_id_for_answer_designation(pred.answer_designation)

    # Ensure predictions are limited to top 25
    predicted_ids = pred.misconception_ids[:prediction_limit]

    # Initialize variables for Average Precision calculation
    score = 0.0

    for rank, pred_id in enumerate(predicted_ids, start=1):
        if pred_id == ground_truth_id:
            precision_at_k = 1 / rank
            score = precision_at_k
            break

    if trace is not None:
        # return score >= 1 / 3  # true if ground_truth_id under the top 3 predicted_ids
        return score == 1

    return score


score_identified_misconceptions_at_25 = partial(score_identified_misconceptions, prediction_limit=25)


def main():

    dspy.settings.configure(
        lm=dspy.LM('openai/gpt-4o-mini'),
    )

    # misconceptions = load_misconceptions(str(mmim_data_path / "misconception_mapping.csv"))
    questions = load_train_data(str(mmim_data_path / "train.csv"))

    # Initialize the pipeline
    identify_misconceptions = IdentifyMisconceptions(
        # num_misconceptions_to_consider=16
        num_misconceptions_to_consider=25
    )

    # # Run the pipeline on the first question
    # result = identify_misconceptions(
    #     question=questions[0]
    # )
    #
    # # Output the results in the required format
    # print(json.dumps(result, indent=4))

    train_examples = []

    for question in questions:
        examples = question.as_dspy_examples()
        train_examples.extend(examples)
        # exampled_dicts = [example.model_dump() for example in examples]
        # train_examples.extend(exampled_dicts)

    logger.critical("Subsampling examples...")
    train_examples = random.Random(9592).sample(train_examples, k=200)

    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=score_identified_misconceptions,
        num_threads=128,  # 16
        # num_threads=256,  # 16
        # max_labeled_demos=16,  # 16
        max_labeled_demos=32,  # 16
        # max_bootstrapped_demos=4,  # 4
        max_bootstrapped_demos=8,  # 4
        # num_candidate_programs=16,  # 16
        num_candidate_programs=24,  # 16
        # num_candidate_programs=32,  # 16
        metric_threshold=1.,
        max_rounds=1,  # 1
    )

    compiled_identify_misconceptions = optimizer.compile(
        identify_misconceptions,
        # teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=train_examples,
    )

    compiled_identify_misconceptions.save(str(PROJECT_ROOT / "out" / "mmim_identify_misconceptions.json"))

    # finetune_config = dict(target=model_to_finetune, epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)
    #
    # # Compile program on BootstrapFinetune
    # finetune_optimizer = BootstrapFinetune(metric=score_identified_misconceptions)
    # finetune_program = finetune_optimizer.compile(
    #     compiled_identify_misconceptions, trainset=train_examples, **finetune_config
    # )
    #
    # finetune_program = your_dspy_program
    #
    # # Load program and activate model's parameters in program before evaluation
    # ckpt_path = "saved_checkpoint_path_from_finetuning"
    # LM = dspy.HFModel(checkpoint=ckpt_path, model=model_to_finetune)
    #
    # for p in finetune_program.predictors():
    #     p.lm = LM
    #     p.activated = False


if __name__ == "__main__":
    main()
