import re
import time
import json
import statistics
from src.pipelines.rag_chain import build_rag_answer
from src.experiments.mlflow_manager import (
    init_mlflow,
    start_run,
    end_run,
    log_params,
    log_metrics
)
from src.utils.logger import logger


def evaluate_single_query(question: str, k: int = 4):
    """
    Executes RAG for a question
    Measures time
    Logs to MLflow
    """

    init_mlflow()

    run = start_run(run_name=f"Eval-{question[:20]}")

    log_params({
        "k_value": k,
        "evaluation_type": "single_query_test"
    })

    try:
        t1 = time.time()
        result = build_rag_answer(question=question, k=k)
        total_time = round(time.time() - t1, 3)

        log_metrics({
            "retrieval_time": result["timing"]["retrieval_time"],
            "generation_time": result["timing"]["generation_time"],
            "total_time": total_time
        })

        logger.info("Evaluation successful")

        end_run(status="FINISHED")

        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        end_run(status="FAILED")
        raise e
    
def compute_retrieval_quality(question: str, docs):
    try:
        question_words = re.findall(r"\b[a-zA-Z]{5,}\b", question.lower())

        if not question_words:
            return 0.0

        doc_text = " ".join([d.page_content.lower() for d in docs])

        matched = 0
        for w in question_words:
            if w in doc_text:
                matched += 1

        score = matched / len(question_words)
        return round(score, 3)

    except Exception as e:
        logger.error(f"Retrieval quality evaluation failed: {str(e)}")
        return 0.0    
    
def evaluate_batch(dataset_path="src/evaluation/eval_questions.json", k: int = 4):
    
    init_mlflow()
    run = start_run(run_name="Batch-Evaluation")

    log_params({
        "k_value": k,
        "evaluation_mode": "batch",
        "dataset": dataset_path
    })

    try:
        with open(dataset_path, "r") as f:
            questions = json.load(f)

        results = []
        total_retrieval_times = []
        total_generation_times = []
        total_total_times = []

        for q in questions:
            question_text = q["question"]

            logger.info(f"Evaluating question: {question_text}")

            result = build_rag_answer(question=question_text, k=k)

            total_retrieval_times.append(result["timing"]["retrieval_time"])
            total_generation_times.append(result["timing"]["generation_time"])
            total_total_times.append(result["timing"]["total_time"])

            results.append({
                "question": question_text,
                "answer_preview": result["answer"][:300],
                "sources": result["sources"],
                "timing": result["timing"]
            })

        avg_retrieval = round(statistics.mean(total_retrieval_times), 3)
        avg_generation = round(statistics.mean(total_generation_times), 3)
        avg_total = round(statistics.mean(total_total_times), 3)

        log_metrics({
            "avg_retrieval_time": avg_retrieval,
            "avg_generation_time": avg_generation,
            "avg_total_time": avg_total,
            "total_questions": len(results)
        })

        with open("batch_eval_results.json", "w") as f:
            json.dump(results, f, indent=4)

        from src.experiments.mlflow_manager import log_artifact
        log_artifact("batch_eval_results.json")

        end_run(status="FINISHED")

        return results

    except Exception as e:
        logger.error(f"Batch evaluation failed: {str(e)}")
        end_run(status="FAILED")
        raise e