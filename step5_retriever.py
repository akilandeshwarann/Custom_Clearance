"""
step5_retriever.py

HybridRetriever: fuses the dense (FAISS/semantic) index from step3_index.py
and the sparse (BM25/keyword) index from step4_index.py using Reciprocal
Rank Fusion (RRF), so queries benefit from both:
  - Dense search: strong on paraphrased / natural-language queries
    ("breeding horses" matches "purebred breeding animals").
  - Sparse search: strong on exact/near-exact token matches, especially
    literal HTS codes ("0101.21.00.10").

RRF combines two ranked lists without needing their raw scores to be on
the same scale (cosine similarity vs. BM25 score aren't comparable
directly) - it only uses each result's *rank position* in each list.

    RRF_score(doc) = sum over each ranking list of  1 / (k + rank(doc))

where `k` is a small constant (60 is the standard default from the
original RRF paper) that dampens the influence of very top ranks.

Inputs (built by earlier steps):
    indexes/faiss_index.bin
    indexes/chunks_metadata.json
    indexes/bm25_index.pkl
    indexes/bm25_chunk_ids.json

Optional filters:
    chapter  - restrict results to a specific HTS chapter (e.g. "01")
    prefix   - restrict results to htsno values starting with a given
               prefix (e.g. "0101" or "0101.21")
"""

import sys
import re
import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from step4_index import tokenize  # reuse the same tokenizer used to build the BM25 index

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60

# Matches a query that IS an HTS code, in whole (e.g. "0101", "0101.21.00.10"),
# as opposed to a query that merely *contains* one among other words.
_FULL_CODE_PATTERN = re.compile(r"^\d{2,4}(\.\d{2}){0,3}$")


def looks_like_hts_code(query):
    return bool(_FULL_CODE_PATTERN.match(query.strip()))


class HybridRetriever:
    def __init__(
        self,
        faiss_path="indexes/faiss_index.bin",
        faiss_metadata_path="indexes/chunks_metadata.json",
        bm25_path="indexes/bm25_index.pkl",
        bm25_chunk_ids_path="indexes/bm25_chunk_ids.json",
        model_name=MODEL_NAME,
    ):
        print("Initializing HybridRetriever...")

        print(f"  Loading FAISS index from '{faiss_path}'...")
        self.faiss_index = faiss.read_index(faiss_path)

        print(f"  Loading FAISS metadata from '{faiss_metadata_path}'...")
        with open(faiss_metadata_path, "r", encoding="utf-8") as f:
            self.faiss_records = json.load(f)

        print(f"  Loading BM25 index from '{bm25_path}'...")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        print(f"  Loading BM25 chunk id mapping from '{bm25_chunk_ids_path}'...")
        with open(bm25_chunk_ids_path, "r", encoding="utf-8") as f:
            self.bm25_records = json.load(f)

        if len(self.faiss_records) != len(self.bm25_records):
            print(
                "Warning: FAISS and BM25 record counts differ "
                f"({len(self.faiss_records)} vs {len(self.bm25_records)}). "
                "Both indexes should be built from the same hts_chunks.json.",
                file=sys.stderr,
            )

        print(f"  Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

        # id -> record, for O(1) lookups when fusing results from both indexes.
        self.records_by_id = {rec["id"]: rec for rec in self.faiss_records}

        print(f"Ready. {len(self.faiss_records)} documents indexed (dense + sparse).")

    # ------------------------------------------------------------------
    # Individual retrieval modes
    # ------------------------------------------------------------------

    def _matches_filters(self, record, chapter=None, prefix=None):
        meta = record.get("metadata", {})
        if chapter is not None and meta.get("chapter") != chapter:
            return False
        if prefix is not None and not str(meta.get("htsno", "")).startswith(prefix):
            return False
        return True

    def dense_search(self, query, top_k=20, chapter=None, prefix=None, min_score=None):
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        # Over-fetch when filters are active, since some hits will be
        # discarded post-hoc (IndexFlatIP has no native metadata filtering).
        search_k = top_k * 5 if (chapter or prefix) else top_k
        search_k = min(search_k, self.faiss_index.ntotal)

        scores, indices = self.faiss_index.search(query_vec, search_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.faiss_records):
                continue
            # For queries that are mostly/only digits and punctuation (e.g. a
            # raw HTS code), the embedding model has little real semantic
            # signal to work with, and cosine similarity can surface
            # coincidentally "nearby" but domain-irrelevant chunks (other
            # numeric-heavy text like CAS numbers or subheading lists). A
            # minimum-similarity floor keeps that noise out of the fused
            # ranking instead of letting it compete with a genuine BM25
            # exact match.
            if min_score is not None and float(score) < min_score:
                continue
            record = self.faiss_records[idx]
            if not self._matches_filters(record, chapter, prefix):
                continue
            results.append((record["id"], float(score), record))
            if len(results) >= top_k:
                break

        return results

    def sparse_search(self, query, top_k=20, chapter=None, prefix=None):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            record = self.bm25_records[idx]
            if not self._matches_filters(record, chapter, prefix):
                continue
            results.append((record["id"], float(scores[idx]), record))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Hybrid retrieval (Reciprocal Rank Fusion)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query,
        top_k=10,
        chapter=None,
        prefix=None,
        rrf_k=RRF_K,
        candidate_k=50,
        weight_dense=1.0,
        weight_sparse=1.0,
        min_dense_score=0.15,
        route_by_query_type=True,
    ):
        """
        Runs dense and sparse search independently, then fuses their
        rankings with (weighted) Reciprocal Rank Fusion:

            score(doc) = weight_dense  * 1/(rrf_k + dense_rank)
                       + weight_sparse * 1/(rrf_k + sparse_rank)

        A document only present in one list still gets a (smaller) score
        from that list alone, so it can still surface in the fused
        results - it just won't get the fusion bonus a doc that ranks
        well in both lists receives.

        `min_dense_score` filters out weak dense hits before fusion, but
        this alone can't fix queries that are themselves a bare HTS code
        (e.g. "0101.21.00.10"): a short, digit/punctuation-heavy query has
        almost no semantic content for the embedding model to work with,
        so even its *top* dense match can score respectably high purely by
        superficial (numeric/formatting) similarity, without being weak
        enough for a threshold to catch - while the actual correct chunk
        may itself score low on the dense side and get filtered out.

        `route_by_query_type`, when True, detects whole-HTS-code queries
        and skips dense search for them entirely, relying purely on BM25's
        exact-token match. This is the more robust fix: routing based on
        query type rather than trying to out-threshold embedding noise.
        """
        use_dense = not (route_by_query_type and looks_like_hts_code(query))

        dense_results = (
            self.dense_search(
                query, top_k=candidate_k, chapter=chapter, prefix=prefix, min_score=min_dense_score
            )
            if use_dense
            else []
        )
        sparse_results = self.sparse_search(query, top_k=candidate_k, chapter=chapter, prefix=prefix)

        rrf_scores = {}
        record_lookup = {}

        for rank, (doc_id, _score, record) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight_dense / (rrf_k + rank)
            record_lookup[doc_id] = record

        for rank, (doc_id, _score, record) in enumerate(sparse_results, start=1):
            # Skip zero-score sparse "matches" - BM25 returns a full score
            # array, and untied-but-irrelevant docs can otherwise still
            # earn a small fusion credit purely from their position in a
            # tie block of genuinely non-matching documents.
            if _score <= 0.0:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight_sparse / (rrf_k + rank)
            record_lookup.setdefault(doc_id, record)

        # Tie-break deterministically on doc_id rather than dict insertion
        # order, so results are reproducible regardless of which retrieval
        # mode happened to run first.
        ranked_ids = sorted(rrf_scores.keys(), key=lambda d: (-rrf_scores[d], d))[:top_k]

        results = []
        for doc_id in ranked_ids:
            record = record_lookup[doc_id]
            results.append({
                "id": doc_id,
                "rrf_score": rrf_scores[doc_id],
                "text": record["text"],
                "metadata": record["metadata"],
            })

        return results


def _print_results(results, label):
    print(f"\n{label}")
    print("-" * 60)
    for rank, r in enumerate(results, start=1):
        meta = r["metadata"] if "metadata" in r else r[2]["metadata"]
        score = r["rrf_score"] if "rrf_score" in r else r[1]
        htsno = meta.get("htsno")
        desc = meta.get("description")
        print(f"Rank {rank} | Score: {score:.4f} | HTS Code: {htsno} | {desc}")
    print("-" * 60)


def main():
    try:
        retriever = HybridRetriever()
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Run step3_index.py and step4_index.py first "
            "to build the FAISS and BM25 indexes.",
            file=sys.stderr,
        )
        sys.exit(1)

    test_queries = [
        "purebred breeding horses import classification",
        "0101.21.00.10",
    ]

    for query in test_queries:
        results = retriever.hybrid_search(query, top_k=5)
        _print_results(results, f"Hybrid search results for: '{query}'")

    # Example of filtered search, restricted to chapter 01
    filtered_results = retriever.hybrid_search(
        "male horses",
        top_k=5,
        chapter="01",
    )
    _print_results(filtered_results, "Hybrid search results for: 'male horses' (chapter=01 filter)")


if __name__ == "__main__":
    main()