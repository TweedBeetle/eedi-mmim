from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Optional, List, Set, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import math


class CorrectAnswerEnum(str, Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'


console = Console()

class Question(BaseModel):
    question_id: int
    construct_id: int
    construct_name: str
    subject_id: int
    subject_name: str
    correct_answer: CorrectAnswerEnum
    question_text: str
    answer_a_text: str
    answer_b_text: str
    answer_c_text: str
    answer_d_text: str

    @field_validator('question_text', 'answer_a_text', 'answer_b_text', 'answer_c_text', 'answer_d_text')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), f"Field cannot be empty"
        return v


class TrainingQuestion(Question):
    misconception_a_id: Optional[int] = Field(default=None)
    misconception_b_id: Optional[int] = Field(default=None)
    misconception_c_id: Optional[int] = Field(default=None)
    misconception_d_id: Optional[int] = Field(default=None)

    @field_validator('misconception_a_id', 'misconception_b_id', 'misconception_c_id', 'misconception_d_id')
    @classmethod
    def handle_nan(cls, v: Optional[float]) -> Optional[int]:
        if v is None or math.isnan(v):
            return None
        return int(v)


class Misconception(BaseModel):
    misconception_id: int
    misconception_name: str

    @field_validator('misconception_name')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), "MisconceptionName cannot be empty"
        return v


class SubmissionEntry(BaseModel):
    QuestionId_Answer: str
    misconception_id: List[int]

    @field_validator('QuestionId_Answer')
    @classmethod
    def validate_format(cls, v: str) -> str:
        parts = v.split('_')
        assert len(parts) == 2, "QuestionId_Answer must be in the format 'QuestionId_AnswerOption'"
        question_id, answer_option = parts
        assert question_id.isdigit(), "QuestionId must be an integer"
        assert answer_option in CorrectAnswerEnum.__members__, "Answer option must be one of A, B, C, D"
        return v

    @field_validator('misconception_id', mode='before')
    @classmethod
    def split_misconceptions(cls, v: str | List[int]) -> List[int]:
        if isinstance(v, str):
            return [int(x) for x in v.strip().split()]
        return v


class MisconceptionPredictionsValidationError(Exception):
    pass


class MisconceptionPredictions(BaseModel):
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
                raise MisconceptionPredictionsValidationError(
                    f"Invalid key '{option}' in 'predicted_misconceptions'. "
                    f"Must be among the incorrect options: {incorrect_options}."
                )

        # Validate misconception IDs
        for option, ids in self.predicted_misconceptions.items():
            if not isinstance(ids, list):
                raise MisconceptionPredictionsValidationError(
                    f"The value for option '{option}' must be a list of integers."
                )
            for mid in ids:
                if not isinstance(mid, int):
                    raise MisconceptionPredictionsValidationError(
                        f"Misconception ID '{mid}' in option '{option}' must be an integer."
                    )

        # Ensure no misconceptions are predicted for the correct answer
        if self.correct_answer in self.predicted_misconceptions:
            raise MisconceptionPredictionsValidationError(
                f"Misconceptions should not be predicted for the correct answer '{self.correct_answer}'."
            )

        return self
