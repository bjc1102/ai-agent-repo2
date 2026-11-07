from importlib.metadata import version

import chromadb
import ollama

from rag_common import QUESTIONS, TOP_K, generate_result, load_chunks, save_results


EMBED_MODEL = "embeddinggemma"


def build_index(chunks):
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="week02-venv",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    texts = [chunk["text"] for chunk in chunks]
    collection.add(
        ids=[f"chunk-{index}" for index in range(len(chunks))],
        documents=texts,
        embeddings=ollama.embed(model=EMBED_MODEL, input=texts, truncate=False).embeddings,
        metadatas=[{"location": chunk["location"]} for chunk in chunks],
    )
    return collection


def search(collection, question):
    result = collection.query(
        query_embeddings=ollama.embed(model=EMBED_MODEL, input=question).embeddings,
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    matches = zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    return [
        {"rank": rank, "location": metadata["location"], "distance": distance, "text": text}
        for rank, (text, metadata, distance) in enumerate(matches, start=1)
    ]


def main():
    chunks = load_chunks()
    collection = build_index(chunks)
    results = [generate_result(question, search(collection, question)) for question in QUESTIONS]
    save_results("results_vector.json", chunks, results, {
        "retrieval": "vector",
        "search_backend": "Chroma",
        "chromadb_version": version("chromadb"),
        "embedding_model": EMBED_MODEL,
        "metric": "cosine",
    })


if __name__ == "__main__":
    main()
