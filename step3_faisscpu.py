"""
step3_index.py

Builds a Dense Vector Index over the HTS chunks using sentence-transformers
(all-MiniLM-L6-v2) and faiss-cpu (IndexFlatIP / cosine similarity via L2
normalization).

Input : hts_chunks.json          (chunked records from step2_chunk.py)
Output: indexes/faiss_index.bin       (binary FAISS index)
        indexes/chunks_metadata.json  (1:1 metadata mapping for vectors)
"""

import sys
import os
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 128


def load_chunks(input_file="hts_chunks.json"):
    print(f"Loading chunks from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    return chunks


def build_faiss_index(
    input_file="hts_chunks.json",
    faiss_output="indexes/faiss_index.bin",
    metadata_output="indexes/chunks_metadata.json",
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
):
    chunks = load_chunks(input_file)
    if not chunks:
        print("Error: no chunks found to index.", file=sys.stderr)
        sys.exit(1)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.shape[1] != EMBEDDING_DIM:
        print(
            f"Warning: expected {EMBEDDING_DIM}-dim embeddings, got "
            f"{embeddings.shape[1]}-dim. Continuing with actual dimension.",
            file=sys.stderr,
        )

    print("Applying L2 normalization (for cosine similarity via inner product)...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built. Total vectors: {index.ntotal}")

    os.makedirs(os.path.dirname(faiss_output) or ".", exist_ok=True)

    print(f"Writing FAISS index to '{faiss_output}'...")
    faiss.write_index(index, faiss_output)

    # chunks_metadata.json is a 1:1 array mapping vector row -> chunk info,
    # so row i in the FAISS index corresponds to chunks_metadata[i].
    metadata_records = [
        {
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
        }
        for chunk in chunks
    ]

    print(f"Writing chunk metadata to '{metadata_output}'...")
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"  FAISS index    : {faiss_output} ({index.ntotal} vectors, dim={dim})")
    print(f"  Metadata file  : {metadata_output} ({len(metadata_records)} records)")

    return index, metadata_records, model


def test_faiss_index(
    index=None,
    metadata_records=None,
    model=None,
    faiss_path="indexes/faiss_index.bin",
    metadata_path="indexes/chunks_metadata.json",
    model_name=MODEL_NAME,
    query="purebred breeding horses import classification",
    top_k=5,
):
    """
    Reloads the index and metadata from disk (unless already provided) and
    runs a sample vector similarity search to sanity-check the build.
    """
    if index is None:
        print(f"Reloading FAISS index from '{faiss_path}'...")
        index = faiss.read_index(faiss_path)

    if metadata_records is None:
        print(f"Reloading chunk metadata from '{metadata_path}'...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_records = json.load(f)

    if model is None:
        print(f"Loading embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)

    print(f"\nRunning test query: '{query}' (top_k={top_k})")
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    print("-" * 60)
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0 or idx >= len(metadata_records):
            continue
        record = metadata_records[idx]
        meta = record.get("metadata", {})
        print(f"Rank {rank} | Score: {score:.4f} | HTS Code: {meta.get('htsno')}")
        print(f"  Description: {meta.get('description')}")
    print("-" * 60)


def main():
    try:
        index, metadata_records, model = build_faiss_index()
    except FileNotFoundError as e:
        print(f"Error: {e}. Run step2_chunk.py first to generate hts_chunks.json.", file=sys.stderr)
        sys.exit(1)

    test_faiss_index(index=index, metadata_records=metadata_records, model=model)


if __name__ == "__main__":
    main()