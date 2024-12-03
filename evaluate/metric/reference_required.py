# Code hasn't been done yet, thanks for visiting anyway
import instructor
from evaluate.utils import CustomGemini
from pydantic import BaseModel
import google.generativeai as genai
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from evaluate.utils import eval_steps


def ref_required_testcase(question: str, response: str, answer: str, level: str):
    """
    Using LLM as a Judge to calculate reference-required metrics.

    Args: question and corresponding reference answer and level from our benchmark, response by LLM generator

    Return: A list of three metrics: Context Relevance, Answer Relevance, Answer Faithfulness
            Each metric contains the score and reason assigned by LLM.

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

    return {
            'metric': 'Correctness',
            'score': correctness_metric.score,
            'reason': correctness_metric.reason
        }

