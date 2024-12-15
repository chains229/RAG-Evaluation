from evaluate.metric.reference_free import ref_free_testcase
from evaluate.metric.reference_required import ref_required_testcase, ref_required_testcase_custom, Remember_Analyze_AnswerTemplate, Understand_AnswerTemplate, Apply_AnswerTemplate, Evaluate_Law_AnswerTemplate, Evaluate_News_AnswerTemplate, Create_AnswerTemplate
import pandas as pd
import statistics as stat

def eval(df, domain: str = "News", llm_judge_name: str = "models/gemini-1.5-pro-002"):
    """
    Evaluate the performance of RAG systems.

    Argument: The pandas dataframe that contains the questions, reference answers from our benchmark,
        as well as the generated response and retrieved context from the RAG system. 

    Return:
        Reference-free metrics: Context Relevance, Answer Relevance, Answer Faithfulness
        Reference-required metrics: As proposed in our paper. 
    """

    columns = ["question", "answer", "response", "relevant_documents", "_id", "level"]
    questions, answers, responses, contexts, ids, levels = (df[col].tolist() for col in columns)

    eval_results = []
    for index in range(len(questions)):
        ref_free_result = ref_free_testcase(questions[index], responses[index], contexts[index])
        ref_required_result = ref_required_testcase_custom(questions[index], responses[index], answers[index], levels[index], llm_judge_name, domain)
        
        print("Done evaluating at index", index)

        eval_results.append({
            "_id": ids[index],
            "con_rel_score": ref_free_result[0]["score"],
            "con_rel_reason": ref_free_result[0]["reason"],
            "ans_rel_score": ref_free_result[1]["score"],
            "ans_rel_reason": ref_free_result[1]["reason"],
            "fai_score": ref_free_result[2]["score"],
            "fai_reason": ref_free_result[2]["reason"],
            "cor_score": ref_required_result["score"],
            "cor_reason": ref_required_result["reason"]
        })
    
    return pd.DataFrame(eval_results)

def eval_avg(df, level, model_name):
    print("________________")
    print(f"Result for model {model_name} at level {level}:")
    
    con_rel_score = stat.mean(df["con_rel_score"].tolist())
    ans_rel_score = stat.mean(df["ans_rel_score"].tolist())
    fai_score = stat.mean(df["fai_score"].tolist())
    cor_score = stat.mean(df["cor_score"].tolist())

    print("Context Relevance Score:", con_rel_score)
    print("Answer Relevance Score:", ans_rel_score)
    print("Faithfulness Score:", fai_score)
    print("Correctness Score:", cor_score)