import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from PyPDF2 import PdfReader
import pandas as pd
import tqdm
from database.utils import *


def create_faiss_index(saved_path, chunk_size: int, chunk_overlap: int, demo: bool, model = "intfloat/multilingual-e5-base", output_folder = "Database"):
    '''
    Retrieve content from CSVs, PDFs and create a FAISS database.
    
    Arguments: 
        input_files: Document files, can be pdf, csv or txt
        github_repo, github_branch, github_folder: Information about the Github folder storing the PDFs
        model: Embedding model, which is intfloat/multilingual-e5-base by default

    Return: Create a FAISS database folder 
    '''
    
    # Load the embedding model
    embedding_model = HuggingFaceEmbeddings(
            model_name=model,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
            show_progress=True,)

    # Process the input documents file
    with open(saved_path, "r", encoding="utf-8") as file:
        documents = file.read()

    documents = chunk_documents(documents, chunk_size, chunk_overlap, tokenizer_name=model, demo=demo)

    print("Done processing files")
    print("_____________________")

    # Create the FAISS vector store
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Save the FAISS index to the output folder
    output_folder_name = output_folder + "_" + str(chunk_size) + "_" + str(chunk_overlap)

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    vector_store.save_local(output_folder_name)

    print(f"FAISS index saved to {output_folder_name}")

    return vector_store


