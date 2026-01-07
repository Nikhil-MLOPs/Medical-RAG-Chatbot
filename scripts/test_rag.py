from src.pipelines.rag_chain import build_rag_chain

chain = build_rag_chain(k=4)

response = chain.invoke("What is diabetes mellitus?")
print("\n----------------- ANSWER -----------------\n")
print(response.content)