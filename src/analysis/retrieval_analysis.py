import asyncio

import openai
from pydantic import BaseModel, Field

from src.models import Question, AnswerDesignation

async_openai_client = openai.AsyncClient()


class MisconceptionAnalysis(BaseModel):
    reasoning: str
    misconception: str = Field(..., description="The misconception that lead to the wrong answer.")


async def identify_misconception(question: Question, wrong_answer_designation: AnswerDesignation):

    formatted_question = f"""Question: {question.format()}"""

    completion = await async_openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Identify the misconception that lead to the wrong answer."},
            {
                "role": "user",
                "content": f"""Decide if the following code looks well documented: {code}"""
            }
        ]
        ,
        response_format=MisconceptionAnalysis,
    )


if __name__ == '__main__':
    asyncio.run(main())
