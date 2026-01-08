import re
import time
import json
import statistics
import numpy as np

from langchain_ollama import OllamaEmbeddings
from src.pipelines.rag_chain import build_rag_answer
from src.pipelines.retrieval import get_retriever

from src.experiments.mlflow_manager import (
    init_mlflow,
    start_run,
    end_run,
    log_params,
    log_metrics,
    log_artifact
)

from src.utils.logger import logger


# -------------------------------
# HELPER – COSINE SIMILARITY
# -------------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -------------------------------
# SINGLE QUERY EVALUATION
# -------------------------------
def evaluate_single_query(question: str, k: int = 4):
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


# -------------------------------
# BATCH EVALUATION
# -------------------------------
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

        # Embedding Model (only once)
        emb = OllamaEmbeddings(model="mxbai-embed-large")
        retriever = get_retriever(k=k)

        results = []

        total_retrieval_times = []
        total_generation_times = []
        total_total_times = []

        retrieval_scores_all = []
        groundedness_scores_all = []
        refusal_flags = []

        for q in questions:
            question_text = q["question"]
            logger.info(f"Evaluating question: {question_text}")

            # -------- Run RAG --------
            result = build_rag_answer(question=question_text, k=k)

            total_retrieval_times.append(result["timing"]["retrieval_time"])
            total_generation_times.append(result["timing"]["generation_time"])
            total_total_times.append(result["timing"]["total_time"])

            # -------- Get Same Docs Used By RAG --------
            docs = retriever.invoke(question_text)

            # -------- Retrieval Relevance Score --------
            q_emb = emb.embed_query(question_text)
            doc_scores = []

            for d in docs:
                doc_emb = emb.embed_query(d.page_content[:500])
                doc_scores.append(cosine(q_emb, doc_emb))

            retrieval_score = round(float(np.mean(doc_scores)), 3)
            retrieval_scores_all.append(retrieval_score)

            # -------- Answer Groundedness Score --------
            ans_emb = emb.embed_query(result["answer"])
            ground_scores = []

            for d in docs:
                doc_emb = emb.embed_query(d.page_content[:500])
                ground_scores.append(cosine(ans_emb, doc_emb))

            groundedness_score = round(float(np.max(ground_scores)), 3)
            groundedness_scores_all.append(groundedness_score)

            # -------- Refusal Compliance --------
            refusal_flag = 1 if (
                "I cannot answer this based on the provided medical reference" in result["answer"]
            ) else 0
            refusal_flags.append(refusal_flag)

            # -------- Store Result --------
            results.append({
                "question": question_text,
                "retrieval_relevance": retrieval_score,
                "groundedness": groundedness_score,
                "refusal_flag": refusal_flag,
                "answer_preview": result["answer"][:250],
                "sources": result["sources"],
                "timing": result["timing"]
            })

        # -------- Averages --------
        avg_retrieval = round(statistics.mean(total_retrieval_times), 3)
        avg_generation = round(statistics.mean(total_generation_times), 3)
        avg_total = round(statistics.mean(total_total_times), 3)

        avg_retrieval_rel = round(statistics.mean(retrieval_scores_all), 3)
        avg_groundedness = round(statistics.mean(groundedness_scores_all), 3)
        refusal_ratio = round(sum(refusal_flags) / len(refusal_flags), 3)

        # -------- Log to MLflow --------
        log_metrics({
            "avg_retrieval_time": avg_retrieval,
            "avg_generation_time": avg_generation,
            "avg_total_time": avg_total,
            "avg_retrieval_relevance": avg_retrieval_rel,
            "avg_answer_groundedness": avg_groundedness,
            "refusal_policy_compliance": refusal_ratio,
            "total_questions": len(results),
        })

        # -------- Save JSON Artifact --------
        with open("batch_eval_results.json", "w") as f:
            json.dump(results, f, indent=4)

        log_artifact("batch_eval_results.json")

        end_run(status="FINISHED")
        return results

    except Exception as e:
        logger.error(f"Batch evaluation failed: {str(e)}")
        end_run(status="FAILED")
        raise e