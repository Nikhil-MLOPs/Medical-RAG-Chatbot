from src.pipelines.ingestion import run_ingestion_pipeline

if __name__ == "__main__":
    chunks = run_ingestion_pipeline()
    print(f"Total chunks ready for embeddings: {len(chunks)}")