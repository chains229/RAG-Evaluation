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
    if args.demo == "true":
        question_df = question_df.head(10)

    print("_____________________________________________________")
    print("Done loading data :3")
    print("_____________________________________________________")

    # RAG pipeline 
    rag_results = rag.test(reader_model_name, faiss_folder, question_df)
    rag_merged_df = question_df.merge(rag_results, on="_id", how="inner")

    print("_____________________________________________________")
    print("Done performing RAG :3")
    print("_____________________________________________________")

    # Evaluation pipeline -> return _id, score
    eval_results = evaluator.eval(rag_merged_df)
    rag_merged_df = rag_merged_df.merge(eval_results, on="_id", how="inner")

    if args.level != "All":
        evaluator.eval_avg(rag_merged_df, args.level, args.model_name)
    else:
        for l in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
            level_df = rag_merged_df[rag_merged_df["level"] == l]
            evaluator.eval_avg(rag_merged_df, args.level, args.model_name)

    print("_____________________________________________________")
    print("Done evaluating :3")
    print("_____________________________________________________")

    # Merge score -> save to a csv
    domain = "news" if "news" in args.data_path else "laws"
    file_name = f"result_{args.level}_{reader_model_name}_{domain}.csv"
    rag_merged_df.to_csv(file_name)

    print("_____________________________________________________")
    print("Done saving results. You can breath now.")
    print("Remember to download the result CSV file")
    print("_____________________________________________________")


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some command-line arguments.')

    # Define the arguments
    parser.add_argument('--model_name', type=str, help="LLM for generator, only support local HuggingFace models")
    parser.add_argument('--data_path', type=str, help="Path to the dataset, can be laws or news domain")
    parser.add_argument('--faiss_folder', type=str, help="Path to the knowledge database")
    parser.add_argument('--level', type=str, choices=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "All"], default="All")
    parser.add_argument('--demo', type=str, choices=['true','false'], help="Run demo, including 10 first rows, to test if code works", default="false")

    args = parser.parse_args()

    main(args)
