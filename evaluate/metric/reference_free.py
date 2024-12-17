from evaluate.utils import CustomGemini
import time
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def ref_free_testcase(question: str, response: str, relevant_docs):
    """
    Using LLM as a Judge to calculate reference-free metrics.

    Args: question from our benchmark, response by LLM generator and relevant_docs from retriever

    Return: A list of three metrics: Context Relevance, Answer Relevance, Answer Faithfulness
            Each metric contains the score and reason assigned by LLM.

    """
    llm_evaluator = CustomGemini()
    con_rel = ContextualRelevancyMetric(model = llm_evaluator, include_reason = True)
    ans_rel = AnswerRelevancyMetric(model = llm_evaluator, include_reason = True)
    faith = FaithfulnessMetric(model = llm_evaluator, include_reason = True)


    testcase = LLMTestCase(
        input = question,
        actual_output = response,
        retrieval_context = relevant_docs
    )

    # Initialize scores list
    scores = []

    # Context Relevance
    try:
        con_rel.measure(testcase)
        scores.append({
            'metric': 'Context Relevance',
            'score': con_rel.score,
            'reason': con_rel.reason
        })
    except Exception as e:
        scores.append({
            'metric': 'Context Relevance',
            'score': 0.0,
            'reason': f'Error: {str(e)}'
        })

    # Answer Relevance
    try:
        ans_rel.measure(testcase)
        scores.append({
            'metric': 'Answer Relevance',
            'score': ans_rel.score,
            'reason': ans_rel.reason
        })
    except Exception as e:
        scores.append({
            'metric': 'Answer Relevance',
            'score': None,
            'reason': f'Error: {str(e)}'
        })

    # Answer Faithfulness
    try:
        faith.measure(testcase)
        scores.append({
            'metric': 'Answer Faithfulness',
            'score': faith.score,
            'reason': faith.reason
        })
    except Exception as e:
        scores.append({
            'metric': 'Answer Faithfulness',
            'score': None,
            'reason': f'Error: {str(e)}'
        })

    return scores

