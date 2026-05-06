"""Golden QA dataset for Ragas evaluation.

These pairs represent ideal questions about RAG concepts — replace or extend
them after ingesting your own domain documents. The ground_truth answers are
used to compute context_recall (does the retrieved context cover the answer?).

To generate a larger synthetic dataset automatically, use LlamaIndex's
`generate_question_context_pairs()` or Ragas' `generate_testset()`.
"""
from typing import Any, Dict, List

GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "ground_truth": (
            "RAG is a technique that combines a retrieval system with a large language model. "
            "It retrieves relevant documents from a knowledge base and uses them as context "
            "to generate accurate, grounded responses, reducing hallucinations."
        ),
    },
    {
        "question": "What are the advantages of hybrid search over pure vector search?",
        "ground_truth": (
            "Hybrid search combines dense semantic vectors with sparse keyword vectors like BM25. "
            "This captures both conceptual similarity and exact keyword matches, outperforming "
            "pure vector search for queries with specific terms, acronyms, or proper nouns."
        ),
    },
    {
        "question": "How does a cross-encoder differ from a bi-encoder for document retrieval?",
        "ground_truth": (
            "A bi-encoder encodes the query and document independently and computes similarity "
            "via dot product — fast but less accurate. A cross-encoder jointly encodes the "
            "query-document pair, yielding much more accurate relevance scores but at higher "
            "computational cost, making it suitable for reranking a small candidate set."
        ),
    },
    {
        "question": "What is Reciprocal Rank Fusion (RRF) and why is it used in RAG?",
        "ground_truth": (
            "RRF is a rank aggregation method that combines ranked lists from multiple retrieval "
            "systems by assigning scores based on the reciprocal of each document's rank. "
            "In RAG, it merges dense and sparse retrieval rankings to produce a unified, "
            "higher-quality candidate list."
        ),
    },
    {
        "question": "What guardrails prevent prompt injection attacks in RAG systems?",
        "ground_truth": (
            "Prompt injection prevention includes input validation, pattern matching against "
            "known injection phrases, LLM-based classifiers, and dialog management frameworks "
            "like NeMo Guardrails that define explicit conversation flows to block adversarial "
            "inputs before they reach the retrieval or generation steps."
        ),
    },
    {
        "question": "What metrics does Ragas use to evaluate RAG systems?",
        "ground_truth": (
            "Ragas measures faithfulness (is the answer grounded in the context?), "
            "answer relevancy (does the answer address the question?), context precision "
            "(are the retrieved chunks relevant to the question?), and context recall "
            "(does the retrieved context cover the ground truth answer?)."
        ),
    },
    {
        "question": "What is the role of BM25 in a hybrid RAG retrieval pipeline?",
        "ground_truth": (
            "BM25 is a probabilistic keyword ranking function that scores documents based on "
            "term frequency and inverse document frequency. In hybrid retrieval, BM25 sparse "
            "vectors capture exact keyword matches that dense embeddings may miss, especially "
            "for rare terms, product codes, or domain-specific jargon."
        ),
    },
    {
        "question": "How does semantic caching improve RAG system performance?",
        "ground_truth": (
            "Semantic caching stores LLM responses keyed by the semantic embedding of the "
            "query. When a new query is semantically similar to a cached one, the stored "
            "response is returned directly, bypassing expensive LLM API calls. This reduces "
            "latency and cost for repeated or paraphrased queries."
        ),
    },
]
