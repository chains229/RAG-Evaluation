import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from transformers import AutoTokenizer
import pandas as pd
import tqdm

# Function to chunk loaded documents
def chunk_documents(documents: str, chunk_size: int, chunk_overlap: int, tokenizer_name, demo):
    """
    Chunk each document's content into smaller pieces and return a new list of documents.
    
    Args:
        documents (str): The document to be splitted.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: List of chunked Document objects.
    """

    RAW_KNOWLEDGE_BASE = []
    documents1 = documents.split("\n\n")
    print("Number of documents:", len(documents1))
    len_doc = len(documents1) if demo == False else int(len(documents1)/10)
    for ind in range(len_doc):
        print("Loading document", ind)
        RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=documents1[ind]))

    NEWS_SEPARATORS = [
        "\n",     # Line breaks
        "\t",     # Tabs
        ". ",     # Sentences
        ", ",     # Clauses within sentences
        "; ",     # Semi-colons separating clauses
        ": ",     # Colons introducing lists or explanations
        " ",      # Spaces
        ""        # No space, as a last resort
        ]
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=NEWS_SEPARATORS,
    )

    docs_processed = []
    for j in range(len(RAW_KNOWLEDGE_BASE)):
        print(j)
        docs_processed += text_splitter.split_documents([RAW_KNOWLEDGE_BASE[j]])

    return docs_processed


# Function to read PDF files
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to read CSV files
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return [" ".join(map(str, row)) for row in df.values]



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