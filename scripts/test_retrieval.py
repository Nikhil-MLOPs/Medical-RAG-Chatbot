from src.pipelines.retrieval import get_retriever

retriever = get_retriever(k=3)

query = "What is diabetes?"
docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} chunks\n")

for i, d in enumerate(docs, 1):
    print(f"------------- Chunk {i} -------------")
    print("Source:", d.metadata.get("source", "N/A"))
    print("Page:", d.metadata.get("page", "N/A"))
    print(d.page_content[:400])
    print()