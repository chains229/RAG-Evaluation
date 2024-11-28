from evaluate.metric.reference_free import ref_free_testcase
from evaluate.metric.reference_required import ref_required_testcase
import pandas as pd

def eval(df):
    """
    Evaluate the performance of RAG systems.

    Argument: The pandas dataframe that contains the questions, reference answers from our benchmark,
        as well as the generated response and retrieved context from the RAG system. 

    Return:
        Reference-free metrics: Context Relevance, Answer Relevance, Answer Faithfulness
        Reference-required metrics: As proposed in our paper. 
    """

    columns = ["question", "answer", "response", "retrieved_context", "_id"]
    questions, answers, responses, contexts, ids = (df[col].tolist() for col in columns)

    eval_results = []
    for index in range(len(questions)):
        ref_free_result = ref_free_testcase(questions[index], responses[index], contexts[index])
        ref_required_result = ref_required_testcase(questions[index], responses[index], answers[index])
        
        
        eval_results.append({
            "_id": ids[index],
            "con_rel_score": ref_free_result[0]["score"],
            "con_rel_reason": ref_free_result[0]["reason"],
            "ans_rel_score": ref_free_result[1]["score"],
            "ans_rel_reason": ref_free_result[1]["reason"],
            "fai_score": ref_free_result[2]["score"],
            "fai_reason": ref_free_result[2]["reason"],
        })
    
    return pd.DataFrame(eval_results)

