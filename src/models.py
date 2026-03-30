from typing import Optional, List, Set, Dict, Tuple

from dspy import Example
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field, ConfigDict
from enum import Enum
import math


class AnswerDesignation(str, Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'


class Question(BaseModel):
    question_id: int = Field(..., alias="QuestionId")
    construct_id: int = Field(..., alias="ConstructId")
    construct_name: str = Field(..., alias="ConstructName")
    subject_id: int = Field(..., alias="SubjectId")
    subject_name: str = Field(..., alias="SubjectName")
    correct_answer: AnswerDesignation = Field(..., alias="CorrectAnswer")
    question_text: str = Field(..., alias="QuestionText")
    answer_a_text: str = Field(..., alias="AnswerAText")
    answer_b_text: str = Field(..., alias="AnswerBText")
    answer_c_text: str = Field(..., alias="AnswerCText")
    answer_d_text: str = Field(..., alias="AnswerDText")

    def designation_for_answer_text(self, answer_text: str) -> AnswerDesignation:
        if answer_text == self.answer_a_text:
            return AnswerDesignation.A
        elif answer_text == self.answer_b_text:
            return AnswerDesignation.B
        elif answer_text == self.answer_c_text:
            return AnswerDesignation.C
        elif answer_text == self.answer_d_text:
            return AnswerDesignation.D
        else:
            raise ValueError(f"Invalid answer text: {answer_text}")

    def answer_text_for_designation(self, answer_designation: AnswerDesignation) -> str:
        if answer_designation == AnswerDesignation.A:
            return self.answer_a_text
        elif answer_designation == AnswerDesignation.B:
            return self.answer_b_text
        elif answer_designation == AnswerDesignation.C:
            return self.answer_c_text
        elif answer_designation == AnswerDesignation.D:
            return self.answer_d_text
        else:
            raise ValueError(f"Invalid answer designation: {answer_designation}")

    @computed_field
    @property
    def all_answers_text(self) -> List[str]:
        return [self.answer_a_text, self.answer_b_text, self.answer_c_text, self.answer_d_text]

    @computed_field
    @property
    def correct_answer_text(self) -> str:
        if self.correct_answer == AnswerDesignation.A:
            return self.answer_a_text
        elif self.correct_answer == AnswerDesignation.B:
            return self.answer_b_text
        elif self.correct_answer == AnswerDesignation.C:
            return self.answer_c_text
        elif self.correct_answer == AnswerDesignation.D:
            return self.answer_d_text
        else:
            raise ValueError(f"Invalid correct answer: {self.correct_answer}")

    @computed_field
    @property
    def incorrect_answers_text(self) -> List[str]:
        return [a for a in self.all_answers_text if a != self.correct_answer_text]

    def __repr__(self) -> str:
        content = (
            f"Question ID: {self.question_id}\n"
            f"Construct: {self.construct_name} (ID: {self.construct_id})\n"
            f"Subject: {self.subject_name} (ID: {self.subject_id})\n"
            f"Question Text: {self.question_text}\n"
            f"Answer Options:\n"
            f"  A) {self.answer_a_text}\n"
            f"  B) {self.answer_b_text}\n"
            f"  C) {self.answer_c_text}\n"
            f"  D) {self.answer_d_text}\n"
            f"Correct Answer: {self.correct_answer}"
        )
        return content

    def __str__(self) -> str:
        content = (
            f"ID: {self.question_id} | "
            f"Construct: {self.construct_name} | "
            f"Subject: {self.subject_name} | "
            f"Correct Answer: {self.correct_answer}\n"
            f"Question: {self.question_text}"
        )
        return content

    def format(self) -> str:
        content = (
            f"Construct: {self.construct_name}\n"
            f"Subject: {self.subject_name}\n"
            f"Question: {self.question_text}\n"
            f"Answer Options:\n"
            f"  A) {self.answer_a_text}\n"
            f"  B) {self.answer_b_text}\n"
            f"  C) {self.answer_c_text}\n"
            f"  D) {self.answer_d_text}\n"
            f"Correct Answer: {self.correct_answer}"
        )
        return content

    @field_validator('question_text', 'answer_a_text', 'answer_b_text', 'answer_c_text', 'answer_d_text')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), f"Field cannot be empty"
        return v


class TrainingQuestion(Question):
    misconception_a_id: Optional[int] = Field(default=None, alias="MisconceptionAId")
    misconception_b_id: Optional[int] = Field(default=None, alias="MisconceptionBId")
    misconception_c_id: Optional[int] = Field(default=None, alias="MisconceptionCId")
    misconception_d_id: Optional[int] = Field(default=None, alias="MisconceptionDId")

    @field_validator('misconception_a_id', 'misconception_b_id', 'misconception_c_id', 'misconception_d_id')
    @classmethod
    def handle_nan(cls, v: Optional[float]) -> Optional[int]:
        if v is None or math.isnan(v):
            return None
        return int(v)

    def as_dspy_examples(self) -> List[Example]:
        examples = []
        for answer_designation in [AnswerDesignation.A, AnswerDesignation.B, AnswerDesignation.C, AnswerDesignation.D]:
            if answer_designation == self.correct_answer:
                continue
            example = Example(
                question=self,
                wrong_answer_designation=answer_designation
            ).with_inputs("question", "wrong_answer_designation")
            examples.append(example)

        return examples

    def misconception_id_for_answer_designation(self, answer_designation: AnswerDesignation) -> int:

        if answer_designation == self.correct_answer:
            raise ValueError("Cannot identify misconceptions for the correct answer.")

        if answer_designation == AnswerDesignation.A:
            return self.misconception_a_id
        elif answer_designation == AnswerDesignation.B:
            return self.misconception_b_id
        elif answer_designation == AnswerDesignation.C:
            return self.misconception_c_id
        elif answer_designation == AnswerDesignation.D:
            return self.misconception_d_id
        else:
            raise ValueError(f"Invalid answer designation: {answer_designation}")


class Misconception(BaseModel):
    misconception_id: int = Field(..., alias="MisconceptionId")
    misconception_name: str = Field(..., alias="MisconceptionName")

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @field_validator('misconception_name')
    @classmethod
    def not_empty(cls, v: str) -> str:
        assert v.strip(), "MisconceptionName cannot be empty"
        return v


class MisconceptionsForAnswer(BaseModel):
    question_id: int = Field(...)
    answer_designation: AnswerDesignation = Field(...)
    misconception_ids: List[int] = Field(...)

    def to_csv_row(self) -> str:
        return f"{self.question_id}_{self.answer_designation}," + " ".join([str(id) for id in self.misconception_ids])

    def items(self) -> List[Tuple[str, List[int]]]:
        return [(str(self.answer_designation), self.misconception_ids)]

# class QuestionMisconceptions(BaseModel):
#     question_id: int = Field(...)
#     misconceptions: Dict[AnswerDesignation, List[int]] = Field(...)
#
#     def to_csv_rows(self) -> List[str]:
#         rows = []
#         for answer_designation, misconception_ids in self.misconceptions.items():
#             rows.append(f"{self.question_id}_{answer_designation}," + " ".join([str(id) for id in misconception_ids]))
#         return rows


class SubmissionEntry(BaseModel):
    question_id: str = Field(..., alias="QuestionId_Answer")
    misconception_id: List[int] = Field(..., alias="MisconceptionId")

    @field_validator('question_id')
    @classmethod
    def validate_format(cls, v: str) -> str:
        parts = v.split('_')
        assert len(parts) == 2, "QuestionId_Answer must be in the format 'QuestionId_AnswerOption'"
        question_id, answer_option = parts
        assert question_id.isdigit(), "QuestionId must be an integer"
        assert answer_option in AnswerDesignation.__members__, "Answer option must be one of A, B, C, D"
        return v

    @field_validator('misconception_id', mode='before')
    @classmethod
    def split_misconceptions(cls, v: str | List[int]) -> List[int]:
        if isinstance(v, str):
            return [int(x) for x in v.strip().split()]
        return v


class MisconceptionPredictionsValidationError(Exception):
    pass


# class MisconceptionPredictions(BaseModel):
#     correct_answer: AnswerDesignation = Field(
#         ...,
#         description="The correct answer option (A, B, C, or D)."
#     )
#     predicted_misconceptions: Dict[str, List[int]] = Field(
#         ...,
#         description=(
#             "Mapping of incorrect answer options to lists of predicted misconception IDs. "
#             "For example: {'B': [101, 202], 'C': [303]}."
#         )
#     )
#
#     @model_validator(mode='after')
#     def validate(self):
#         valid_options: Set[str] = {'A', 'B', 'C', 'D'}
#         incorrect_options = valid_options - {self.correct_answer}
#
#         # Validate keys
#         for option in self.predicted_misconceptions.keys():
#             if option not in incorrect_options:
#                 raise MisconceptionPredictionsValidationError(
#                     f"Invalid key '{option}' in 'predicted_misconceptions'. "
#                     f"Must be among the incorrect options: {incorrect_options}."
#                 )
#
#         # Validate misconception IDs
#         for option, ids in self.predicted_misconceptions.items():
#             if not isinstance(ids, list):
#                 raise MisconceptionPredictionsValidationError(
#                     f"The value for option '{option}' must be a list of integers."
#                 )
#             for mid in ids:
#                 if not isinstance(mid, int):
#                     raise MisconceptionPredictionsValidationError(
#                         f"Misconception ID '{mid}' in option '{option}' must be an integer."
#                     )
#
#         # Ensure no misconceptions are predicted for the correct answer
#         if self.correct_answer in self.predicted_misconceptions:
#             raise MisconceptionPredictionsValidationError(
#                 f"Misconceptions should not be predicted for the correct answer '{self.correct_answer}'."
#             )
#
#         return self


if __name__ == '__main__':
    question = Question(
        question_id=1,
        construct_id=101,
        construct_name="Linear Equations",
        subject_id=10,
        subject_name="Mathematics",
        correct_answer=AnswerDesignation.A,
        question_text="Solve for x: 2x + 3 = 7",
        answer_a_text="x = 2",
        answer_b_text="x = 1",
        answer_c_text="x = -2",
        answer_d_text="x = 0"
    )

    # print(repr(question))
    print(question.format())
    # print(str(question))
