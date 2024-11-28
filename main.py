import argparse
import RAG.RAG as rag
import evaluate.evaluator as evaluator
import pandas as pd

def main(args):
    reader_model_name = args.model_name
    faiss_folder = args.faiss_folder

    # Load data
    question_df = pd.read_csv(args.data_path)
    if args.level != "All":
        question_df = question_df[question_df['level'] == args.level]

    print("Done loading data :3")
    print("_________________________")

    # RAG pipeline 
    rag_results = rag.test(reader_model_name, faiss_folder, question_df)
    rag_merged_df = question_df.merge(rag_results, on="_id", how="inner")

    print("Done performing RAG :3")
    print("_________________________")

    # Evaluation pipeline -> return _id, score
    # eval_results = evaluator.eval(rag_merged_df)

    # print("Done evaluating :3")
    # print("_________________________")

    # Merge score -> save to a csv
    domain = "news" if "news" in args.data_path else "laws"
    file_name = f"result_{args.level}_{reader_model_name}_{domain}.csv"
    rag_merged_df.to_csv(file_name)

    print("Done saving results. You can breath now.")
    print("Remember to download the result CSV file")
    print("_________________________")


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some command-line arguments.')

    # Define the arguments
    parser.add_argument('--model_name', type=str, help="LLM for generator, only support local HuggingFace models")
    parser.add_argument('--data_path', type=str, help="Path to the dataset, can be laws or news domain")
    parser.add_argument('--faiss_folder', type=str, help="Path to the knowledge database")
    parser.add_argument('--level', type=str, choices=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "All"], default="All")


    args = parser.parse_args()

    main(args)
