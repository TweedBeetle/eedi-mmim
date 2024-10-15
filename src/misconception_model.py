from pydantic import BaseModel, Field, root_validator, model_validator
from typing import Dict, List, Set
from enum import Enum

from src.models.frozen_base_model import FrozenBaseModel
from src.models.tasking.evaluation_error import EvaluationError


class CorrectAnswerEnum(str, Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'


class MisconceptionsModel(FrozenBaseModel):
    correct_answer: CorrectAnswerEnum = Field(
        ...,
        description="The correct answer option (A, B, C, or D)."
    )
    predicted_misconceptions: Dict[str, List[int]] = Field(
        ...,
        description=(
            "Mapping of incorrect answer options to lists of predicted misconception IDs. "
            "For example: {'B': [101, 202], 'C': [303]}."
        )
    )

    @model_validator(mode='after')
    def validate(self):
        valid_options: Set[str] = {'A', 'B', 'C', 'D'}
        incorrect_options = valid_options - {self.correct_answer}

        # Validate keys
        for option in self.predicted_misconceptions.keys():
            if option not in incorrect_options:
                raise EvaluationError(
                    f"Invalid key '{option}' in 'predicted_misconceptions'. "
                    f"Must be among the incorrect options: {incorrect_options}."
                )

        # Validate misconception IDs
        for option, ids in self.predicted_misconceptions.items():
            if not isinstance(ids, list):
                raise EvaluationError(f"The value for option '{option}' must be a list of integers.")
            for mid in ids:
                if not isinstance(mid, int):
                    raise EvaluationError(f"Misconception ID '{mid}' in option '{option}' must be an integer.")

        # Ensure no misconceptions are predicted for the correct answer
        if self.correct_answer in self.predicted_misconceptions:
            raise EvaluationError(f"Misconceptions should not be predicted for the correct answer '{self.correct_answer}'.")

        return self
