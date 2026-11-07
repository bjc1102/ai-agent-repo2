import re
import sqlite3
from contextlib import closing

from rag_common import QUESTIONS, TOP_K, generate_result, load_chunks, save_results


def search(connection, question):
    # Separate English words from Korean particles, such as "Python에".
    terms = re.findall(r"[A-Za-z0-9]+|[가-힣]+", question)
    if not terms:
        return []

    # Match any question term and rank the matching chunks with BM25.
    query = " OR ".join(f'"{term}"' for term in terms)
    rows = connection.execute(
        """
        SELECT text, location, bm25(chunks) AS score
        FROM chunks
        WHERE chunks MATCH ?
        ORDER BY score, rowid
        LIMIT ?
        """,
        (query, TOP_K),
    )
    return [
        {"rank": rank, "location": location, "bm25_score": score, "text": text}
        for rank, (text, location, score) in enumerate(rows, start=1)
    ]


def main():
    chunks = load_chunks()
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5(text, location UNINDEXED, tokenize='unicode61')"
        )
        connection.executemany(
            "INSERT INTO chunks (text, location) VALUES (:text, :location)", chunks
        )
        results = [generate_result(question, search(connection, question)) for question in QUESTIONS]
    save_results("results_keyword.json", chunks, results, {
        "retrieval": "keyword",
        "search_backend": "SQLite FTS5",
        "sqlite_version": sqlite3.sqlite_version,
        "metric": "bm25",
        "tokenizer": "unicode61",
        "query_tokenization": "separate English and Korean terms",
        "query_operator": "OR",
    })


if __name__ == "__main__":
    main()
