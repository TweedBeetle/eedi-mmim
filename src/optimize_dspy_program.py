import argparse
import json
from typing import List, Dict, Optional

import dspy
from dspy import Module, Signature, InputField, OutputField, ChainOfThought, Predict
from dspy.retrieve.weaviate_rm import WeaviateRM
from loguru import logger
from tqdm import tqdm

from src.models import Misconception, CorrectAnswerEnum
from src.my_weaviate import get_weaviate_client
from src.data_loader import load_misconceptions, load_train_data
from src.constants import mmim_data_path
import ollama


# Define Signatures

class GenerateMisconceptionQuerySignature(Signature):
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


# Define Modules

class GenerateMisconceptionQuery(Module):
    """Describe the misconception that lead to the wrong answer."""

    def __init__(self):
        super().__init__()
        self.signature = GenerateMisconceptionQuerySignature()
        self.chain = ChainOfThought(self.signature)

    def forward(self, question: str, answer_option: str) -> str:
        response = self.chain(question=question, answer_option=answer_option)
        return response.misconception_query


class RetrieveMisconceptions(Module):
    """Module to retrieve misconceptions from Weaviate using DSPy's WeaviateRM."""

    def __init__(
            self,
            weaviate_collection_name: str = "Misconception",
            weaviate_collection_text_key: str = "misconception_name",
            k: int = 5
    ):
        super().__init__()
        self.signature = RetrieveMisconceptionsSignature()
        # Initialize DSPy's WeaviateRM retrieval model
        self.retriever = WeaviateRM(
            weaviate_collection_name=weaviate_collection_name,
            weaviate_client=get_weaviate_client(),
            weaviate_collection_text_key=weaviate_collection_text_key,
            k=k  # Number of top results to retrieve
        )

    def forward(self, query: str) -> List[Misconception]:
        query = "search_query: " + query

        # Use the built-in retrieval
        retrieved = self.retriever(query=query)
        misconceptions = []
        for obj in retrieved.objects:
            misconception = Misconception(
                misconception_id=obj.properties['misconception_id'],
                misconception_name=obj.properties['misconception_name']
            )
            misconceptions.append(misconception)
        return misconceptions


class ClassifyMisconception(Module):
    """Classify whether the wrong answer to the question is due to the given misconception."""

    def __init__(self):
        super().__init__()
        self.signature = ClassifyMisconceptionSignature()
        self.predictor = Predict(self.signature)

    def forward(self, question: str, misconception: Misconception) -> bool:
        response = self.predictor(question=question, misconception=misconception)
        answer = response.include.strip().lower()
        return answer == 'true'


# Define the Main Pipeline

class MMIMPipeline(Module):
    """Multi-Hop Misconception Identification and Management Pipeline."""

    def __init__(self, num_misconceptions_to_consider: int = 25):
        super().__init__()
        self.generate_misconception_query = GenerateMisconceptionQuery()
        self.retrieve_misconceptions = RetrieveMisconceptions()
        self.classify_misconception = ClassifyMisconception()
        self.num_misconceptions_to_consider = num_misconceptions_to_consider

    def forward(self, question: str, correct_answer: str, all_answers: Dict[str, str]) -> Dict[str, List[int]]:
        included_misconceptions = {}

        # Identify incorrect answer options
        incorrect_options = [opt for opt in all_answers.keys() if opt != correct_answer]

        for option in incorrect_options:
            logger.info(f"Processing Misconceptions for Incorrect Answer Option: {option}")

            # Step 1: Generate Misconception Query
            misconception_query = self.generate_misconception_query(question=question, answer_option=option)
            logger.debug(
                f"Generated Misconception Query for Option {option}: {misconception_query}"
            )  # @todo:0: generate multiple misconception descriptions (eg as a numbered list)

            # Step 2: Retrieve Top-k Misconceptions
            misconceptions = self.retrieve_misconceptions(misconception_query=misconception_query)
            logger.debug(f"Retrieved {len(misconceptions)} Misconceptions for Option {option}")

            # Step 3: Classify Each Misconception
            for misconception in misconceptions:
                logger.info(f"Classifying Misconception ID {misconception.misconception_id} for Option {option}")
                include = self.classify_misconception(question=question, misconception=misconception)
                logger.debug(f"Include Misconception {misconception.misconception_id}: {include}")
                if include:
                    included_misconceptions.setdefault(option, []).append(misconception.misconception_id)
                    logger.info(f"Included Misconception ID {misconception.misconception_id} for Option {option}")

        return included_misconceptions


# Optimizer Configuration (Optional)

def main():

    # misconceptions = load_misconceptions(str(mmim_data_path / "misconception_mapping.csv"))
    questions = load_train_data(str(mmim_data_path / "train.csv"))

    # Initialize the pipeline
    pipeline = MMIMPipeline(
        num_misconceptions_to_consider=25
    )

    # Run the pipeline on the first question
    result = pipeline(
        question=args.question,
        correct_answer=args.correct_answer,
        all_answers=all_answers
    )

    # Output the results in the required format
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
