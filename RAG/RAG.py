import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Pipeline
from typing import List, Tuple
import torch
import json
import tqdm
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as LangchainDocument
from RAG.utils import load_reader_model, create_prompt_template

def answer_one_sample(question: str, llm: Pipeline, knowledge_index: FAISS,
    rag_prompt_template, num_retrieved_docs: int = 10,) -> Tuple[str, List[LangchainDocument]]:
    """
    Function to answer a question with RAG
    """
    
    # Retrieve documents
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs_content = [doc.page_content for doc in relevant_docs]

    # Build the context from retrieved documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc + "\n" for i, doc in enumerate(relevant_docs_content)])

    # Build the final prompt
    final_prompt = rag_prompt_template.format(question=question, context=context)

    # Generate an answer
    answer = llm(final_prompt)[0]["generated_text"]

    # Clear GPU memory
    torch.cuda.empty_cache()

    return answer, relevant_docs


def test(reader_model_name: str, faiss_folder: str, questions_df, topk):
    """
    Run RAG pipeline on our benchmark.

    Arguments:
        reader_model_name: Local model as the generator
        faiss_folder: Path to the database folder
        questions_df: DataFrame of the CSV file containing the questions in our benchmark.

    Return: DataFrame of results, including questions, answers and retrieved contexts.
    """
    
    # Initialize components
    llm = load_reader_model(reader_model_name)

    embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
            show_progress=True,)

    knowledge_index = FAISS.load_local(
        faiss_folder, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    rag_prompt_template = create_prompt_template(AutoTokenizer.from_pretrained(reader_model_name))

    # Load questions from dataframe
    questions = questions_df["question"].tolist()
    ids = questions_df["_id"].tolist()

    # Process each question and collect answers
    results = []
    for i in range(len(questions)):
        response, relevant_docs = answer_one_sample(
            questions[i], llm, knowledge_index, rag_prompt_template, num_retrieved_docs=topk
        )
        results.append({
            "_id": ids[i],
            "response": response,
            "retrieved_context": [doc.page_content for doc in relevant_docs]
        })
        print(questions[i])
        print(relevant_docs)
        print(response)

    return pd.DataFrame(results)