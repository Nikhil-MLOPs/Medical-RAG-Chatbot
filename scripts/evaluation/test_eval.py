from src.evaluation.evaluator import evaluate_single_query

if __name__ == "__main__":
    result = evaluate_single_query(
        question="What is diabetes?",
        k=4
    )

    print(result["answer"])
