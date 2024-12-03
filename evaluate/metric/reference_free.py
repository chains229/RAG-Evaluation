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

    con_rel_score = con_rel.measure(testcase)
    ans_rel_score = ans_rel.measure(testcase)
    fai_score = faith.measure(testcase)

    return [
        {
            'metric': 'Context Relevance',
            'score': con_rel_score.score,
            'reason': con_rel_score.reason
        },
        {
            'metric': 'Answer Relevance',
            'score': ans_rel_score.score,
            'reason': ans_rel_score.reason
        },
        {
            'metric': 'Answer Faithfulness',
            'score': fai_score.score,
            'reason': fai_score.reason
        }
    ]

