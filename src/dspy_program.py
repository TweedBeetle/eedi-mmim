import argparse
import concurrent
import json
from typing import List, Dict, Optional

import dspy
from dspy import Module, Signature, InputField, OutputField, ChainOfThought, Predict, Example
from dspy.datasets import Dataset
from dspy.retrieve.weaviate_rm import WeaviateRM
from loguru import logger
from tqdm import tqdm

from src.models import Misconception, AnswerDesignation, Question, MisconceptionsForAnswer
from src.my_weaviate import get_weaviate_client, retrieve_misconceptions
from src.data_loading import load_misconceptions, load_train_data
from src.constants import mmim_data_path
import ollama


# Define Signatures

class DescribeMisconceptionSignature(Signature):
    """Describe the misconception that lead to the wrong answer."""
    question = InputField(description="The original mathematical question.")
    right_answer = InputField(description="The correct answer given to the question.")
    wrong_answer = InputField(description="The incorrect answer given to the question.")
    misconception_description = OutputField(
        description="A single sentence description of the misconception that lead to the wrong answer."
    )


class RetrieveMisconceptionsSignature(Signature):
    """Retrieve top-k misconceptions based on the misconception query."""
    misconception_query = InputField(description="The search query to retrieve misconceptions.")
    misconceptions = OutputField(description="List of retrieved Misconception objects.")


class ClassifyMisconceptionSignature(Signature):
    """Decide whether the misconception is an underlying reason for the wrong answer to the question."""
    question = InputField(description="The original mathematical question.")
    right_answer = InputField(description="The correct answer given to the question.")
    wrong_answer = InputField(description="The incorrect answer that was chosen.")
    misconception = InputField(
        description="The misconception to consider as the potential underlying reason for the wrong answer."
    )

    misconception_underlies_wrong_answer = OutputField(
        description="Boolean indicating whether the misconception is a underlying reason for the wrong answer."
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

    # def forward(self, question: str, correct_answer: str, all_answers: Dict[str, str]) -> Dict[str, List[int]]:
    # def forward(self, question: Question) -> QuestionMisconceptions:
    def forward(self, question: Question, wrong_answer_designation: AnswerDesignation) -> MisconceptionsForAnswer:
        identified_misconceptions = []

        wrong_answer_text = question.answer_text_for_designation(wrong_answer_designation)

        logger.info(f"Processing Misconceptions for Incorrect Answer Option: {wrong_answer_text}")

        # Step 1: Generate Misconception Query
        misconception_description = self.generate_misconception_query(
            question=question.question_text,
            right_answer=question.correct_answer_text,
            wrong_answer=wrong_answer_text,
        ).completions[0].misconception_description

        logger.debug(
            f"Generated Misconception description for Option {wrong_answer_text}: {misconception_description}"
        )  # @todo:0: generate multiple misconception descriptions (eg as a numbered list)

        # Step 2: Retrieve Top-k Misconceptions
        misconceptions = self.retrieve_misconceptions(
            query_or_queries=misconception_description,

        )

        logger.debug(f"Retrieved {len(misconceptions)} Misconceptions for Option {wrong_answer_text}")

        # Step 3: Classify Each Misconception
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_misconception = {
                executor.submit(
                    self.classify_misconception,
                    question=question.question_text,
                    right_answer=question.correct_answer_text,
                    wrong_answer=wrong_answer_text,
                    misconception=misconception.misconception_name
                ): misconception for misconception in misconceptions
            }

            for future in concurrent.futures.as_completed(future_to_misconception):
                misconception = future_to_misconception[future]
                try:
                    result = future.result()
                    misconception_underlies_wrong_answer = result.completions[
                        0].misconception_underlies_wrong_answer

                    logger.debug(
                        f"Include Misconception {misconception.misconception_id}: {misconception_underlies_wrong_answer}"
                    )

                    if misconception_underlies_wrong_answer:
                        identified_misconceptions.append(
                            misconception.misconception_id
                        )
                        logger.debug(
                            f"Included Misconception ID {misconception.misconception_id} for Option {wrong_answer_text}"
                        )
                except Exception as exc:
                    logger.error(
                        f"Error classifying Misconception ID {misconception.misconception_id} for Option {wrong_answer_text}: {exc}"
                    )

        return MisconceptionsForAnswer(
            question_id=question.question_id,
            answer_designation=wrong_answer_designation,
            misconception_ids=identified_misconceptions
        )


def score_identified_misconceptions(
        example,
        pred: MisconceptionsForAnswer,
) -> float:
    # example.question: Question
    # example.wrong_answer_designation: AnswerDesignation

    raise NotImplementedError()


def main():

    dspy.settings.configure(
        lm=dspy.LM('openai/gpt-4o-mini'),
    )

    # misconceptions = load_misconceptions(str(mmim_data_path / "misconception_mapping.csv"))
    questions = load_train_data(str(mmim_data_path / "train.csv"))

    # Initialize the pipeline
    identify_misconceptions = IdentifyMisconceptions(
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
        train_examples.extend(question.as_dspy_examples())

    optimizer = dspy.BootstrapFewShot(metric=score_identified_misconceptions)
    compiled_identify_misconceptions = optimizer.compile(
        identify_misconceptions,
        # teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=train_examples,
    )


if __name__ == "__main__":
    main()
