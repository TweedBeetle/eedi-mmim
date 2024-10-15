# Example Input Data
from src.sidequests.eedi_mmim.src.misconception_model import CorrectAnswerEnum
from src.sidequests.eedi_mmim.src.mmim_task import MMIMTask

input_data = MMIMTask.Input(
    question_id=1,
    subject_id=101,
    subject_name="Mathematics",
    construct_id=201,
    construct_name="Algebra",
    question_text="Solve for x: 2x + 3 = 7",
    answers={
        "A": "x = 1",
        "B": "x = 2",
        "C": "x = 3",
        "D": "x = 4"
    },
    correct_answer=CorrectAnswerEnum.B,
    ground_truth_misconceptions={
        "A": [101],
        "C": [102],
        "D": [103]
    }
)

# Example Output Data (from the LLM or tasker)
output_data = MMIMTask.Output(
    question_understanding="This question tests the ability to solve linear equations.",
    correct_answer_analysis="The correct answer is x = 2. Solving 2x + 3 = 7 gives x = 2.",
    incorrect_options_analysis={
        "A": "Mistake in subtracting 3 from both sides.",
        "C": "Forgot to divide by 2 after subtracting 3.",
        "D": "Added instead of subtracting."
    },
    context_consideration="Students often make errors in basic algebraic manipulations.",
    misconception_reasoning="Each incorrect option represents a common error.",
    predicted_misconceptions={
        "A": [201],
        "C": [202],
        "D": [203]
    }
)

# Create an MMIMTask instance
task = MMIMTask(input=input_data, output=output_data)

# Get the final result
final_result = task.result_in_final_form

# This will be a Misco
