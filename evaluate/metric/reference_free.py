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
    time.sleep(5)
    ans_rel = AnswerRelevancyMetric(model = llm_evaluator, include_reason = True)
    time.sleep(5)
    faith = FaithfulnessMetric(model = llm_evaluator, include_reason = True)
    time.sleep(5)

    testcase = LLMTestCase(
        input = question,
        actual_output = response,
        retrieval_context = relevant_docs
    )

    con_rel.measure(testcase)
    ans_rel.measure(testcase)
    faith.measure(testcase)

    scores = [
        {
            'metric': 'Context Relevance',
            'score': con_rel.score,
            'reason': con_rel.reason
        },
        {
            'metric': 'Answer Relevance',
            'score': ans_rel.score,
            'reason': ans_rel.reason
        },
        {
            'metric': 'Answer Faithfulness',
            'score': faith.score,
            'reason': faith.reason
        }
    ]

    return scores

