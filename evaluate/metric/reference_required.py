# Code hasn't been done yet, thanks for visiting anyway
import instructor
from evaluate.utils import CustomGemini, prompt
from pydantic import BaseModel
import google.generativeai as genai
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from evaluate.utils import eval_steps
import time
import typing

class answer_template(typing.TypedDict):
    score: int
    reason: dict

def ref_required_testcase_custom(question: str, response: str, answer: str, level: str, llm_judge_name: str):
    """
    Using LLM as a Judge to calculate reference-required metrics using our custom prompt.

    Args: question and corresponding reference answer and level from our benchmark, response by LLM generator

    Return: Correctness score and its details provided by LLM
    """
    model = genai.GenerativeModel(llm_judge_name)

    correctness_metric = model.generate_content(
        prompt(level, question, response, answer),
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=answer_template), temperature = 0.0)
    cor_score = {
            'metric': 'Correctness',
            'score': correctness_metric.score,
            'reason': correctness_metric.reason
        }
    return cor_score



def ref_required_testcase(question: str, response: str, answer: str, level: str):
    """
    Using LLM as a Judge to calculate reference-required metrics using G-Eval prompt.

    Args: question and corresponding reference answer and level from our benchmark, response by LLM generator

    Return: Correctness score and reason provided by LLM

    """
    llm_evaluator = CustomGemini()
    correctness_metric = GEval(
        name="Correctness",
        model=llm_evaluator,
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        evaluation_steps=eval_steps(level),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        expected_output=answer
    )

    correctness_metric.measure(test_case)

    cor_score = {
            'metric': 'Correctness',
            'score': correctness_metric.score,
            'reason': correctness_metric.reason
        }
    return cor_score