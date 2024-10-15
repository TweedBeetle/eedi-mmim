import json
import random
from functools import cache
from typing import ClassVar, Type, List, Dict, Optional
from pydantic import Field, root_validator
from loguru import logger
from tqdm import tqdm

from src.models.tasking.task import Task, UnevaluatedCompletedTask
from src.models.tasking.task_input import TaskInput
from src.models.tasking.task_output import TaskOutput
from src.models.tasking.evaluation import Evaluation
from src.models.tasking.tasker import Tasker
from src.models.data_split import DataSplit
from src.my_wandb.decorators import logged_wandb_run
from src.my_wandb.wandb_artifact import WandbArtifactBirth, WandbArtifactVessel
from src.sidequests.eedi_mmim.src.data_loader import load_train_data, load_misconception_mapping
from src.sidequests.eedi_mmim.src.mmim_constants import mmim_data_path
from src.sidequests.eedi_mmim.src.models import TrainModel
from src.sidequests.eedi_mmim.src.misconception_model import MisconceptionsModel, CorrectAnswerEnum


@cache
def get_formatted_misconceptions():
    misconceptions = load_misconception_mapping(str(mmim_data_path / "misconception_mapping.csv"))
    formatted_misconceptions = "\n".join([f"{m.misconception_id}: {m.misconception_name}" for m in misconceptions])
    return f"Available misconceptions:\n{formatted_misconceptions}"


class MMIMTask(
    Task['MMIMTask.Input', 'MMIMTask.Output']
):
    @classmethod
    def get_instructions(cls) -> str:
        return (
            "Analyze the given mathematics question with multiple-choice answers to predict the misconceptions "
            "associated with each incorrect answer option. Provide a comprehensive analysis of the question, "
            "correct answer, and incorrect options, considering the mathematical context and potential student "
            "misunderstandings. Use your analysis to identify and justify the most probable misconceptions for "
            "each incorrect answer. Choose from the provided list of available misconceptions."
        )

    class Input(TaskInput):
        question_id: int = Field(..., description="Unique identifier for the question.", repr=False)
        question_text: str = Field(..., description="The text of the question.")
        correct_answer: str = Field(..., description="The correct answer option (A, B, C, or D).")
        answers: Dict[str, str] = Field(
            ...,
            description=(
                "A dictionary of answer options, mapping option letters to answer texts. "
                "For example: {'A': 'Option A text', 'B': 'Option B text', 'C': 'Option C text', 'D': 'Option D text'}"
            )
        )
        construct_id: int = Field(
            ..., description="The construct ID representing the mathematical concept being tested.", repr=False
        )
        construct_name: str = Field(..., description="The name of the mathematical construct being tested.")
        subject_id: int = Field(..., description="The subject ID.", repr=False)
        subject_name: str = Field(..., description="The name of the subject.")
        ground_truth_misconceptions: Optional[Dict[str, List[int]]] = Field(
            None,
            description=(
                "Optional ground truth misconceptions mapping each incorrect answer option "
                "to a list of misconception IDs. For example: {'A': [101, 202], 'C': [303]}."
            ),
            repr=False  # not given to the LLM, just used for evaluation
        )
        available_misconceptions: str = Field(
            default_factory=get_formatted_misconceptions,
            description="A formatted string containing all available misconceptions to choose from."
        )

        def __hash__(self):
            return hash(self.question_id)

        def __eq__(self, other):
            if not isinstance(other, MMIMTask.Input):
                return NotImplemented
            return self.question_id == other.question_id

        @classmethod
        def get_field_order(cls) -> List[str]:
            return [
                'question_id',
                'subject_id',
                'subject_name',
                'construct_id',
                'construct_name',
                'question_text',
                'answers',
                'correct_answer',
                'ground_truth_misconceptions',
                'available_misconceptions',
            ]

        @classmethod
        def from_question(cls, question: TrainModel) -> 'MMIMTask.Input':
            answers = {
                'A': question.answer_a_text,
                'B': question.answer_b_text,
                'C': question.answer_c_text,
                'D': question.answer_d_text
            }

            ground_truth_misconceptions = {}
            for option, misconception_id in [('A', question.misconception_a_id),
                                             ('B', question.misconception_b_id),
                                             ('C', question.misconception_c_id),
                                             ('D', question.misconception_d_id)]:
                if misconception_id is not None:
                    ground_truth_misconceptions[option] = [misconception_id]

            return cls(
                question_id=question.question_id,
                question_text=question.question_text,
                correct_answer=question.correct_answer,
                answers=answers,
                construct_id=question.construct_id,
                construct_name=question.construct_name,
                subject_id=question.subject_id,
                subject_name=question.subject_name,
                ground_truth_misconceptions=ground_truth_misconceptions if ground_truth_misconceptions else None
            )

    input_type: ClassVar[Type[Input]] = Input

    class Output(TaskOutput):
        question_understanding: str = Field(
            ...,
            description=(
                "Provide a comprehensive and meticulous analysis of the question.\n\n"
                "**Instructions:**\n"
                "1. **Detailed Question Analysis:**\n"
                "   - Examine the mathematical concept being tested in depth.\n"
                "   - Discuss specific skills or knowledge required to answer correctly.\n"
                "   - Identify potential areas of difficulty for students.\n"
                "   - Analyze how the question's wording or presentation might influence understanding.\n"
                "2. **Guidelines:**\n"
                "   - Be thorough and methodical, ensuring no aspect is overlooked.\n"
                "   - Avoid drawing conclusions about misconceptions at this stage.\n"
                "   - Provide detailed observations, starting from the most obvious and moving to subtle nuances.\n"
                "   - Use numbered lists to organize your observations when appropriate."
            )
        )

        correct_answer_analysis: str = Field(
            ...,
            description=(
                "Analyze the correct answer in detail.\n\n"
                "**Instructions:**\n"
                "1. **Detailed Correct Answer Analysis:**\n"
                "   - Explain why this answer is correct, including key steps or insights needed to arrive at this answer.\n"
                "   - Discuss how understanding these steps relates to the tested concept.\n"
                "2. **Guidelines:**\n"
                "   - Be detailed and methodical.\n"
                "   - Avoid considering misconceptions at this point.\n"
                "   - Use clear explanations to illuminate the reasoning behind the correct answer."
            )
        )

        incorrect_options_analysis: str = Field(
            ...,
            description=(
                "Provide a comprehensive analysis of all incorrect answer options.\n\n"
                "**Instructions:**\n"
                "1. **Individual Incorrect Option Analysis:**\n"
                "   - For each incorrect answer option, discuss:\n"
                "     - How students might arrive at this incorrect answer.\n"
                "     - Potential misunderstandings or errors in reasoning leading to this choice.\n"
                "     - Common mistakes in this topic area that might be relevant.\n"
                "     - How the option might seem plausible to students with certain misconceptions.\n"
                "   - Note any patterns or relationships between the incorrect options.\n"
                "2. **Guidelines:**\n"
                "   - Analyze each incorrect option separately and in depth.\n"
                "   - Avoid prematurely assigning specific misconceptions.\n"
                "   - Use evidence-based reasoning, drawing on knowledge of common student errors."
            )
        )

        context_consideration: str = Field(
            ...,
            description=(
                "Analyze the broader context of the question.\n\n"
                "**Instructions:**\n"
                "1. **Contextual Analysis:**\n"
                "   - Discuss how the subject area (e.g., algebra, geometry) might influence likely misconceptions.\n"
                "   - Explore how the specific mathematical construct being tested relates to common misconceptions.\n"
                "   - Consider relevant developmental stages or typical cognitive challenges for the target age group.\n"
                "2. **Guidelines:**\n"
                "   - Provide a thorough analysis of contextual factors.\n"
                "   - Reference educational theories or frameworks where appropriate.\n"
                "   - Be methodical and avoid assumption-based reasoning."
            )
        )

        initial_misconception_reasoning: str = Field(
            ...,
            description=(
                "Provide detailed reasoning for identifying an extensive list of potential misconceptions.\n\n"
                "**Instructions:**\n"
                "1. **Comprehensive Misconception Exploration:**\n"
                "   - Reference common mathematical errors and conceptual misunderstandings relevant to the question.\n"
                "   - Evaluate the plausibility of each potential misconception in relation to this specific question and its incorrect options.\n"
                "   - Consider multiple misconceptions that might apply to incorrect answers.\n"
                "   - Assess which misconceptions are more fundamental or likely.\n"
                "   - Cross-check identified misconceptions against the question and answers to ensure consistency.\n"
                "   - Consider any potential contradictions or mutually exclusive misconceptions.\n"
                "2. **Guidelines:**\n"
                "   - Aim to identify at least **100 potential misconceptions**.\n"
                "   - Be exhaustive and thorough in your reasoning.\n"
                "   - Do not narrow down or exclude misconceptions at this stage.\n"
                "   - Use numbered lists to organize misconceptions where appropriate."
            )
        )

        top_100_misconceptions: str = Field(
            ...,
            description=(
                "List the **top 100 most likely misconceptions** that could be associated with the incorrect answers.\n\n"
                "**Instructions:**\n"
                "1. **Listing Misconceptions:**\n"
                "   - Provide a numbered list of 100 misconception IDs and names from the available misconceptions.\n"
                "   - Ensure the list is based on your initial detailed reasoning.\n"
                "2. **Guidelines:**\n"
                "   - Do not prioritize or rank the misconceptions beyond including them in this list.\n"
                "   - Do not assign misconceptions to specific answer options yet.\n"
                "   - Ensure all listed misconceptions are plausible based on your analysis."
            )
        )

        intermediate_misconception_reasoning: str = Field(
            ...,
            description=(
                "Provide detailed reasoning to narrow down the list of misconceptions to the **top 25 most likely** ones.\n\n"
                "**Instructions:**\n"
                "1. **Evaluation and Narrowing Down:**\n"
                "   - Re-examine the top 100 misconceptions in light of the question and incorrect options.\n"
                "   - Discuss the relevance and applicability of each misconception.\n"
                "   - Identify misconceptions that are most strongly connected to the question and incorrect answers.\n"
                "   - Consider eliminating misconceptions with less direct relevance.\n"
                "2. **Guidelines:**\n"
                "   - Be thorough and justify the inclusion or exclusion of misconceptions.\n"
                "   - Use evidence from your previous analyses to support your reasoning.\n"
                "   - Do not assign misconceptions to specific answer options yet."
            )
        )

        top_25_misconceptions: str = Field(
            ...,
            description=(
                "List the **top 25 most likely misconceptions** after your intermediate analysis.\n\n"
                "**Instructions:**\n"
                "1. **Listing Misconceptions:**\n"
                "   - Provide a numbered list of the 25 misconception IDs and names.\n"
                "   - Ensure the list reflects the misconceptions most relevant to the question.\n"
                "2. **Guidelines:**\n"
                "   - Do not assign misconceptions to specific incorrect options yet.\n"
                "   - Ensure the list is based on your intermediate reasoning."
            )
        )

        final_misconception_reasoning: str = Field(
            ...,
            description=(
                "Provide a detailed analysis and thought process for each incorrect answer option to assign the final misconceptions.\n\n"
                "**Instructions:**\n"
                "1. **Option-Specific Analysis:**\n"
                "   - For each incorrect answer option:\n"
                "     - Examine the mathematical principles and potential misunderstandings leading to this choice.\n"
                "     - Consider how the **top 25 misconceptions** specifically relate to this option.\n"
                "     - Analyze the option in the context of common student errors and the question's wording.\n"
                "     - Explore multiple possible explanations for why a student might choose this option.\n"
                "     - Compare and contrast different misconceptions that could apply.\n"
                "   - Justify the selection of the most appropriate misconceptions for each incorrect answer.\n"
                "2. **Guidelines:**\n"
                "   - Be thorough and methodical in your reasoning.\n"
                "   - Ensure that the final selected misconceptions align with your analyses.\n"
                "   - Limit the number of misconceptions per incorrect answer as appropriate.\n"
                "   - Use evidence-based reasoning to support your selections."
            )
        )

        final_misconceptions: str = Field(
            ...,
            description=(
                "List the final selected misconceptions for each incorrect answer option.\n\n"
                "**Instructions:**\n"
                "1. **Final Misconception Assignment:**\n"
                "   - For each incorrect answer option, list the misconception IDs selected based on your final reasoning.\n"
                "   - Provide a brief summary or justification if necessary.\n"
                "2. **Guidelines:**\n"
                "   - Ensure that the selected misconceptions are consistent with your previous analyses.\n"
                "   - Be clear and concise in your presentation."
            )
        )

        final_formatted_misconceptions: str = Field(
            ...,
            description=(
                "From the final misconceptions, provide a **JSON string** representing a dictionary mapping each incorrect answer option to a list of predicted misconception IDs.\n\n"
                "For example: `'{\"A\": [101, 202], \"C\": [303]}'` for options A and C.\n\n"
                "**Ensure that:**\n"
                "• The JSON is valid and properly formatted.\n"
                "• The selected misconceptions match your `final_misconceptions` field.\n"
                "• Each prediction is consistent with your `final_misconception_reasoning`."
            )
        )

        @classmethod
        def get_field_order(cls) -> List[str]:
            return [
                'question_understanding',
                'correct_answer_analysis',
                'incorrect_options_analysis',
                'context_consideration',
                'initial_misconception_reasoning',
                'top_100_misconceptions',
                'intermediate_misconception_reasoning',
                'top_25_misconceptions',
                'final_misconception_reasoning',
                'final_misconceptions',
                'final_formatted_misconceptions',
            ]

    output_type: ClassVar[Type[Output]] = Output

    @classmethod
    async def evaluate_completed_task(
            cls,
            completed_task: 'UnevaluatedCompletedTask[Input, Output]',
    ) -> Evaluation:
        """
        Evaluate the completed task by comparing the predicted misconceptions with the ground truth when available.
        If the ground truth is not provided, return a passed evaluation by default.
        """
        input_data = completed_task.input

        misconceptions_model = completed_task.result_in_final_form

        if input_data.ground_truth_misconceptions:
            gt_misconceptions = input_data.ground_truth_misconceptions
            predicted = misconceptions_model.predicted_misconceptions

            total_options = len(gt_misconceptions)
            correct_predictions = 0
            detailed_feedback = ""

            for option, gt_ids in gt_misconceptions.items():
                pred_ids = set(predicted.get(option, []))
                gt_ids_set = set(gt_ids)

                if pred_ids == gt_ids_set:
                    correct_predictions += 1
                else:
                    missing = gt_ids_set - pred_ids
                    extra = pred_ids - gt_ids_set
                    feedback = f"Option '{option}': "
                    if missing:
                        feedback += f"Missing misconceptions {missing}. "
                    if extra:
                        feedback += f"Unjustified misconceptions {extra}. "
                    detailed_feedback += feedback + "\n"

            accuracy = correct_predictions / total_options if total_options else 0

            if accuracy == 1.0:
                return Evaluation.create_passed()
            else:
                return Evaluation.create_failed(
                    feedback=f"Misconception predictions did not fully match the ground truth.\n{detailed_feedback}",
                    label="MISCONCEPTIONS_MISMATCH"
                )
        else:
            return Evaluation.create_passed()

    @property
    def result_in_final_form(self) -> MisconceptionsModel:
        """
        Return the predicted misconceptions as a MisconceptionModel instance.
        """

        raw_final_formatted_misconceptions = self.output.final_formatted_misconceptions

        # return MisconceptionsModel.model_validate_json(raw_final_formatted_misconceptions)

        return MisconceptionsModel(
            correct_answer=self.input.correct_answer,
            predicted_misconceptions=json.loads(raw_final_formatted_misconceptions)
        )

    @classmethod
    def get_default_tasker(cls) -> Tasker:
        """
        Return the default Tasker for this task.
        You can customize this based on your taskers (e.g., OpenAI GPT-based tasker).
        """
        return Tasker()  # Replace with your actual default Tasker

    @classmethod
    @logged_wandb_run()
    def birth_initial_input_data_split(
            cls
    ) -> WandbArtifactBirth[DataSplit['MMIMTask.Input']]:
        """
        Creates the initial data split for MMIMTask inputs using the training data.
        We do not expect any failures; raises an exception if any occur.
        """
        # Load training data
        train_filepath = mmim_data_path / "train.csv"
        train_data: List[TrainModel] = load_train_data(str(train_filepath))
        logger.info(f"Loaded {len(train_data)} training records")

        # Convert each record to MMIMTask.Input
        inputs: List[MMIMTask.Input] = []
        for record in tqdm(train_data, desc="Converting training records to MMIMTask.Input"):
            input_instance = cls.Input.from_question(record)
            inputs.append(input_instance)

        logger.info(f"Successfully created {len(inputs)} MMIMTask.Input instances")

        # Shuffle inputs for randomness
        rng = random.Random(42)  # Fixed seed for reproducibility
        rng.shuffle(inputs)

        # Create DataSplit
        InputDataSplit = cls.get_input_data_split_class()
        data_split = InputDataSplit.from_items(inputs, split=(0.8, 0.1, 0.1))
        logger.info(
            f"Created data split with {len(data_split.train)} train, {len(data_split.val)} val, and {len(data_split.test)} test instances"
        )

        # logger.critical("subsmapling datasplit. remove this")
        # data_split = data_split.sample(0.01)

        # Return as a WandbArtifactBirth
        return WandbArtifactBirth[InputDataSplit](
            artifact=data_split,
            metadata={
                'num_inputs': len(inputs),
            }
        )


MMIMTask.model_rebuild()
