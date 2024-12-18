from evaluate.utils import CustomGemini
import time
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from concurrent.futures import ThreadPoolExecutor, as_completed

def ref_free_testcase(question: str, response: str, relevant_docs):
    """
    Efficiently calculate reference-free metrics using concurrent execution.
    """
    llm_evaluator = CustomGemini()
    
    # Create metrics outside of try-except to reduce overhead
    metrics = [
        ('Context Relevance', ContextualRelevancyMetric(model=llm_evaluator, include_reason=True)),
        ('Answer Relevance', AnswerRelevancyMetric(model=llm_evaluator, include_reason=True)),
        ('Answer Faithfulness', FaithfulnessMetric(model=llm_evaluator, include_reason=True))
    ]
    
    testcase = LLMTestCase(
        input=question,
        actual_output=response,
        retrieval_context=relevant_docs
    )
    
    # Use ThreadPoolExecutor for concurrent metric calculation
    scores = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Create a future for each metric
        future_to_metric = {
            executor.submit(metric_obj.measure, testcase): metric_name 
            for metric_name, metric_obj in metrics
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_metric):
            metric_name = future_to_metric[future]
            try:
                # The measure method already updates the metric object
                metric_obj = next(m for name, m in metrics if name == metric_name)
                scores.append({
                    'metric': metric_name,
                    'score': metric_obj.score,
                    'reason': metric_obj.reason
                })
            except Exception as e:
                scores.append({
                    'metric': metric_name,
                    'score': None,
                    'reason': f'Error: {str(e)}'
                })
    
    return scores

