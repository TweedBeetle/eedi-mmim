from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import math

from src.models.frozen_base_model import FrozenBaseModel


from src.sidequests.eedi_mmim.src.misconception_model import CorrectAnswerEnum


class TrainModel(FrozenBaseModel):
    QuestionId: int
    ConstructId: int
    ConstructName: str
    SubjectId: int
    SubjectName: str
    CorrectAnswer: CorrectAnswerEnum
    QuestionText: str
    AnswerAText: str
    AnswerBText: str
    AnswerCText: str
    AnswerDText: str
    MisconceptionAId: Optional[int] = Field(default=None)
    MisconceptionBId: Optional[int] = Field(default=None)
    MisconceptionCId: Optional[int] = Field(default=None)
    MisconceptionDId: Optional[int] = Field(default=None)

    @field_validator('QuestionText', 'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), f"Field cannot be empty"
        return v

    @field_validator('MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId')
    @classmethod
    def handle_nan(cls, v: Optional[float]) -> Optional[int]:
        if v is None or math.isnan(v):
            return None
        return int(v)


class TestModel(FrozenBaseModel):
    QuestionId: int
    ConstructId: int
    ConstructName: str
    SubjectId: int
    SubjectName: str
    CorrectAnswer: CorrectAnswerEnum
    QuestionText: str
    AnswerAText: str
    AnswerBText: str
    AnswerCText: str
    AnswerDText: str

    @field_validator('QuestionText', 'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), f"Field cannot be empty"
        return v


class MisconceptionMappingModel(FrozenBaseModel):
    MisconceptionId: int
    MisconceptionName: str

    @field_validator('MisconceptionName')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), "MisconceptionName cannot be empty"
        return v


class SampleSubmissionModel(FrozenBaseModel):
    QuestionId_Answer: str
    MisconceptionId: List[int]

    @field_validator('QuestionId_Answer')
    @classmethod
    def validate_format(cls, v: str) -> str:
        parts = v.split('_')
        assert len(parts) == 2, "QuestionId_Answer must be in the format 'QuestionId_AnswerOption'"
        question_id, answer_option = parts
        assert question_id.isdigit(), "QuestionId must be an integer"
        assert answer_option in CorrectAnswerEnum.__members__, "Answer option must be one of A, B, C, D"
        return v

    @field_validator('MisconceptionId', mode='before')
    @classmethod
    def split_misconceptions(cls, v: str | List[int]) -> List[int]:
        if isinstance(v, str):
            return [int(x) for x in v.strip().split()]
        return v
