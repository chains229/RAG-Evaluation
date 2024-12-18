# Code hasn't been done yet, thanks for visiting anyway
import instructor
from evaluate.utils import CustomGemini
from evaluate.metric.template import prompt
from pydantic import BaseModel
import google.generativeai as genai
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from evaluate.utils import eval_steps
import time
from typing_extensions import TypedDict
import json
import sys
sys.set_int_max_str_digits(10000)

class Remember_Analyze_AnswerTemplate(TypedDict):
    accuracy_score: int
    accuracy_justification: str

class Understand_AnswerTemplate(TypedDict):
    content_coverage_score: int
    content_coverage_justification: str
    information_accuracy_score: int
    information_accuracy_justification: str
    paraphrasing_quality_score: int
    paraphrasing_quality_justification: str
    overall_similarity_score: int
    overall_similarity_justification: str

class Apply_AnswerTemplate(TypedDict):
    application_of_knowledge_score: int
    application_of_knowledge_justification: str
    completeness_score: int
    completeness_justification: str
    accuracy_score: int
    accuracy_justification: str
    practicality_score: int
    practicality_justification: str

class Evaluate_News_AnswerTemplate(TypedDict):
    perspective_diversity_score: int
    perspective_diversity_justification: str
    dispute_awareness_score: int
    dispute_awareness_justification: str
    alignment_with_partial_answers_score: int
    alignment_with_partial_answers_justification: str
    
class Evaluate_Law_AnswerTemplate(TypedDict):
    accuracy_score: int
    accuracy_justification: str
    reasoning_similarity_score: int
    reasoning_similarity_justification: str

class Create_AnswerTemplate(TypedDict):
    essay_structure_score: float
    essay_structure_justification: str
    identification_of_argumentative_issue_score: float
    identification_of_argumentative_issue_justification: str
    development_of_argument_score: float
    development_of_argument_justification: str
    spelling_and_grammar_score: float
    spelling_and_grammar_justification: str
    creativity_score: float
    creativity_justification: str

# Dictionary mapping levels to their respective templates
LEVEL_TO_TEMPLATE = {
    "Remember": Remember_Analyze_AnswerTemplate,
    "Understand": Understand_AnswerTemplate,
    "Apply": Apply_AnswerTemplate,
    "Analyze": Remember_Analyze_AnswerTemplate,
    "Evaluate_News": Evaluate_News_AnswerTemplate,
    "Evaluate_Law": Evaluate_Law_AnswerTemplate,
    "Create": Create_AnswerTemplate
}

def ref_required_testcase_custom(question: str, response: str, answer: str, level: str, llm_judge_name: str, domain: str):
    """
    Using LLM as a Judge to calculate reference-required metrics using our custom prompt.

    Args: question and corresponding reference answer and level from our benchmark, response by LLM generator

    Return: Correctness score and its details provided by LLM
    """
    model = genai.GenerativeModel(llm_judge_name)
    if level == "Evaluate":
        l = level + "_" + domain
    else:
        l = level
    answer_template = LEVEL_TO_TEMPLATE[l]
    
    try:
        responsed_metric = model.generate_content(
                contents = prompt(level, question, response, answer, domain),
                generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=answer_template, temperature = 1.0,max_output_tokens = 2048))
    
        response = json.loads(responsed_metric.text)
        print(response)

        average_score = calculate_average_score(response, l)
    
        cor_score = {
                    'metric': 'Correctness',
                    'score': average_score,
                    'reason': response
                }
        print("Correctness score:", average_score)
        return cor_score
    except:
        cor_score = {
            'metric': 'Correctness',
            'score': 0.0,
            'reason': "Failed to transform output to json. Details:" + responsed_metric.text
        }
        print(cor_score)
        return cor_score
    
    

def calculate_average_score(responsed_metric: dict, level: str) -> float:
    """
    Calculate the average score of all metrics in a given level.

    Args:
        responsed_metric (dict): The dictionary containing scores for the level.
        level (str): The cognitive level for evaluation.

    Returns:
        float: The average score across all metrics in the level.
    """
    # Define the fields to extract scores for each level
    level_fields = {
        "Remember_Analyze": ["accuracy_score"],
        "Understand": [
            "content_coverage_score",
            "information_accuracy_score",
            "paraphrasing_quality_score",
            "overall_similarity_score"
        ],
        "Apply": [
            "application_of_knowledge_score",
            "completeness_score",
            "accuracy_score",
            "practicality_score"
        ],
        "Evaluate_News": [
            "perspective_diversity_score",
            "dispute_awareness_score",
            "alignment_with_partial_answers_score"
        ],
        "Evaluate_Law": [
            "accuracy_score",
            "reasoning_similarity_score"
        ],
        "Create": [
            "essay_structure_score",
            "identification_of_argumentative_issue_score",
            "development_of_argument_score",
            "spelling_and_grammar_score",
            "creativity_score"
        ]
    }

    if level in "Remember_Analyze":
        scores = 1 if responsed_metric["accuracy_score"] > 0 else 0
        return float(scores)
    elif level == "Evaluate_Law" or level == "Create":
        # scores = [responsed_metric[field] for field in level_fields[level]]
        # return sum(scores) if scores else 0.0
        scores = [value for key, value in responsed_metric.items() if "score" in key and isinstance(value, (int, float))]        
        return sum(scores) / 10 if scores else 0.0
    scores = [value for key, value in responsed_metric.items() if "score" in key and isinstance(value, (int, float))]        
    return sum(scores) / (len(scores)*10) if scores else 0.0

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