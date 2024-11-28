import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from PyPDF2 import PdfReader
import pandas as pd
import tqdm
from database.utils import *

# Function to process input files and convert them into documents
def process_files(file_paths):
    documents = []
    for file_path in tqdm(file_paths):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                text = read_pdf(file_path)
                documents.append(Document(page_content=text))
            elif ext == ".csv":
                rows = read_csv(file_path)
                for row in rows:
                    documents.append(Document(page_content=row))
            else:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    documents.append(Document(page_content=text))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return documents

def create_faiss_index(input_csvs: list, input_pdfs: list, model = "intfloat/multilingual-e5-base", output_folder = "Database"):
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

    input_files = input_csvs + input_pdfs

    # Process the input files into documents
    documents = process_files(input_files)

    print("Done processing files")
    print("_____________________")

    # Create the FAISS vector store
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Save the FAISS index to the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    vector_store.save_local(output_folder)

    print(f"FAISS index saved to {output_folder}")

    return vector_store


