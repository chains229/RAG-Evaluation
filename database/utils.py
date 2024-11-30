import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import pandas as pd
import tqdm

# Function to read PDF files
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\\n"
    return text

# Function to read CSV files
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return [" ".join(map(str, row)) for row in df.values]

# Function to chunk loaded documents
def chunk_documents(documents, chunk_size, chunk_overlap):
    """
    Chunk each document's content into smaller pieces and return a new list of documents.
    
    Args:
        documents (list): List of Document objects.
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Overlap between chunks.
    
    Returns:
        list: List of chunked Document objects.
    """
    full_doc = ""
    for doc in documents:
        full_doc += doc + "\n"
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_doc = splitter.split_text(full_doc)

    chunked_documents = []
    for doc in splitted_doc:
        chunked_documents.append(Document(page_content=doc, metadata=doc.metadata))
    print(f"Chunked documents into {len(chunked_documents)} total chunks.")
    return chunked_documents

# No need to read two below functions

# Function to download a file from a URL
def download_file(url, output_dir="downloads"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    local_filename = os.path.join(output_dir, url.split("/")[-1])
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            f.write(response.content)
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

# Function to get all file URLs in a GitHub folder
def get_github_folder_files(repo, branch, folder_path):
    api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}?ref={branch}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        files = response.json()
        return [
            f"https://raw.githubusercontent.com/{repo}/{branch}/{file['path']}"
            for file in files if file['name'].endswith('.pdf')
        ]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching files from GitHub: {e}")
        return []