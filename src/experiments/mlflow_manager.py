import os
import mlflow
from src.utils.logger import logger


MLFLOW_EXPERIMENT_NAME = "Medical-RAG-Evaluation"


def init_mlflow():
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow initialized with SQLite backend.")
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {str(e)}")
        raise e


def start_run(run_name: str):
    """
    Starts a new MLflow run.
    """

    try:
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow Run Started: {run.info.run_id}")
        return run

    except Exception as e:
        logger.error(f"Failed to start MLflow run: {str(e)}")
        raise e


def end_run(status: str = "FINISHED"):
    """
    Ends MLflow run with status.
    """

    try:
        mlflow.end_run(status=status)
        logger.info(f"MLflow Run Ended with status: {status}")

    except Exception as e:
        logger.error(f"Failed to end MLflow run: {str(e)}")
        raise e


def log_params(params: dict):
    try:
        mlflow.log_params(params)
    except Exception as e:
        logger.error(f"Failed logging params: {str(e)}")


def log_metrics(metrics: dict):
    try:
        mlflow.log_metrics(metrics)
    except Exception as e:
        logger.error(f"Failed logging metrics: {str(e)}")


def log_artifact(file_path: str):
    try:
        mlflow.log_artifact(file_path)
    except Exception as e:
        logger.error(f"Failed logging artifact: {str(e)}")
