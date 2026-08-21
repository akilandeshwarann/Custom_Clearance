"""
step4_index.py

Builds a BM25 sparse/keyword index over the HTS chunks using rank_bm25.
This complements the dense FAISS index from step3_index.py: BM25 is much
stronger at exact/near-exact token matching, which matters a lot here
because users frequently search by literal HTS code (e.g. "0101.21.00.10")
rather than natural language, and dense embeddings alone tend to blur
exact numeric codes together.

Input : hts_chunks.json          (chunked records from step2_chunk.py)
Output: indexes/bm25_index.pkl   (pickled BM25Okapi index + tokenized corpus)
        indexes/bm25_chunk_ids.json (row -> chunk id / htsno mapping, 1:1 with the index)
"""

import sys
import os
import re
import json
import pickle

from rank_bm25 import BM25Okapi

# Matches HTS codes (e.g. "0101", "0101.21.00", "0101.21.00.10") as single
# tokens first, so they don't get shredded into separate numeric fragments.
# Falls back to plain words and standalone numbers otherwise.
_TOKEN_PATTERN = re.compile(r"\d{2,4}(?:\.\d{2}){1,3}|\d{4,}|[A-Za-z]+")


def tokenize(text):
    """
    Custom tokenizer that preserves HTS codes intact (so a query for
    '0101.21.00.10' matches the exact code token) while lowercasing and
    splitting ordinary description text into plain words.
    """
    if not text:
        return []
    tokens = _TOKEN_PATTERN.findall(text)
    normalized = []
    for tok in tokens:
        # Leave HTS-code-shaped tokens (contain a dot) as-is; lowercase words.
        normalized.append(tok if "." in tok else tok.lower())
    return normalized


def load_chunks(input_file="hts_chunks.json"):
    print(f"Loading chunks from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    return chunks


def build_bm25_index(
    input_file="hts_chunks.json",
    bm25_output="indexes/bm25_index.pkl",
    chunk_ids_output="indexes/bm25_chunk_ids.json",
):
    chunks = load_chunks(input_file)
    if not chunks:
        print("Error: no chunks found to index.", file=sys.stderr)
        sys.exit(1)

    print(f"Tokenizing {len(chunks)} chunks...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(bm25_output) or ".", exist_ok=True)

    print(f"Writing BM25 index to '{bm25_output}'...")
    with open(bm25_output, "wb") as f:
        pickle.dump(bm25, f)

    # Row i in the BM25 index corresponds to chunk_ids[i]; this mirrors the
    # FAISS metadata mapping from step3_index.py so step5_retriever.py can
    # fuse results from both indexes using a shared row -> chunk lookup.
    chunk_id_records = [
        {
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
        }
        for chunk in chunks
    ]

    print(f"Writing chunk id mapping to '{chunk_ids_output}'...")
    with open(chunk_ids_output, "w", encoding="utf-8") as f:
        json.dump(chunk_id_records, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"  BM25 index      : {bm25_output} ({len(tokenized_corpus)} documents)")
    print(f"  Chunk id mapping: {chunk_ids_output} ({len(chunk_id_records)} records)")

    return bm25, chunk_id_records


def test_bm25_index(
    bm25=None,
    chunk_id_records=None,
    bm25_path="indexes/bm25_index.pkl",
    chunk_ids_path="indexes/bm25_chunk_ids.json",
    query="0101.21.00.10",
    top_k=5,
):
    """
    Reloads the BM25 index and chunk id mapping from disk (unless already
    provided) and runs a sample keyword search to sanity-check the build.
    Defaults to an exact HTS code query, since that's BM25's core strength
    relative to the dense index.
    """
    if bm25 is None:
        print(f"Reloading BM25 index from '{bm25_path}'...")
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

    if chunk_id_records is None:
        print(f"Reloading chunk id mapping from '{chunk_ids_path}'...")
        with open(chunk_ids_path, "r", encoding="utf-8") as f:
            chunk_id_records = json.load(f)

    print(f"\nRunning test query: '{query}' (top_k={top_k})")
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    print("-" * 60)
    for rank, idx in enumerate(ranked_indices, start=1):
        record = chunk_id_records[idx]
        meta = record.get("metadata", {})
        print(f"Rank {rank} | Score: {scores[idx]:.4f} | HTS Code: {meta.get('htsno')}")
        print(f"  Description: {meta.get('description')}")
    print("-" * 60)


def main():
    try:
        bm25, chunk_id_records = build_bm25_index()
    except FileNotFoundError as e:
        print(f"Error: {e}. Run step2_chunk.py first to generate hts_chunks.json.", file=sys.stderr)
        sys.exit(1)

    test_bm25_index(bm25=bm25, chunk_id_records=chunk_id_records)


if __name__ == "__main__":
    main()