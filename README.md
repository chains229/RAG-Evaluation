# **Evaluating RAG Systems**

Our code produces emperical results in our research, which basically proposed a benchmark in evaluating RAG systems using Revised Bloom's Taxonomy. We don't even know if we can ever publish the paper.


# How to run
Use our notebook (.ipynb) file (i will update i promise), maybe you will understand all the code and instructions I wrote. I hope it will work. The arguments you can control when running the notebook are:
- chunk_size, chunk_overlap: Chunk size and overlap size in chunking phase.
- embedding_model: The embedding model to generate knowledge database.
- Data source: Include initial documents and question-answer files.
- model_name: The LLM generator model.
- topk: The number of retrieved documents
- level: The specific level in Revised Bloom's Taxonomy that you want to evaluate. Besides six levels of the taxonomy, you can set it to "All" to evaluate all levels.
- demo: I use this to check if the code works lol


# To-do
- Add argument: chunking technique (from langchain.text_splitter)
- Create a GUI using Streamlit for more user-friendly evaluation process.