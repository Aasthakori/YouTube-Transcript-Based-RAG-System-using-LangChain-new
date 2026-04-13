"""Gemini-judged RAGAS evaluation of the RAG pipeline across 3 videos.

Run as:
    python -m src.evaluation.gemini_eval

Flow:
    1. Load evaluation/eval_dataset.json (30 questions, 10 per video).
    2. For each video: delete index → re-ingest → run full pipeline for all 10 Qs.
    3. Save all 30 answers to evaluation/results/gemini_answers.json.
    4. Run RAGAS (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference)
       using Gemini Flash as the judge — answerable questions only (24 total).
    5. String-match refusal phrases on unanswerable questions (6 total).
    6. Print per-video + average scores and refusal accuracy.
    7. Save results to evaluation/results/gemini_eval_results.json.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)

from src.config import FAISS_INDEX_PATH
from src.generation.citations import _REFUSAL_PHRASES
from src.generation.chains import _get_llm
from src.generation.citations import format_with_sources
from src.generation.prompts import rag_prompt
from src.indexing.vector_store import create_vector_store, load_index, save_index
from src.ingestion.chunker import chunk_transcript
from src.ingestion.youtube import fetch_transcript, get_video_title
from src.retrieval.hybrid import build_hybrid_retriever
from src.retrieval.reranker import build_reranking_retriever

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = Path("evaluation/eval_dataset.json")
ANSWERS_PATH = Path("evaluation/results/gemini_answers.json")
RESULTS_PATH = Path("evaluation/results/gemini_eval_results.json")

QUESTIONS_PER_VIDEO = 10

# Human-readable labels in file order
VIDEO_LABELS = ["Vidya Balan", "Kriti Sanon", "Shahid Kapoor"]

# Normalize RAGAS result keys → our internal names
_RAGAS_KEY_MAP: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "response_relevancy",   # older RAGAS versions
    "response_relevancy": "response_relevancy",  # newer RAGAS versions
    "llm_context_precision_without_reference": "context_precision",
}


# ---------------------------------------------------------------------------
# Section 1 — Load & group questions
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[list[dict[str, Any]]]:
    """Load the eval dataset and split into groups of QUESTIONS_PER_VIDEO.

    Questions are grouped by position in the file, not by metadata, so
    unanswerable rows (which lack expected_video_id) stay with their video.

    Args:
        path: Path to the JSON eval dataset file.

    Returns:
        List of groups, each containing QUESTIONS_PER_VIDEO question dicts.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file is empty or not divisible by QUESTIONS_PER_VIDEO.
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    with path.open() as f:
        rows: list[dict[str, Any]] = json.load(f)
    if not rows:
        raise ValueError("Eval dataset is empty.")
    if len(rows) % QUESTIONS_PER_VIDEO != 0:
        raise ValueError(
            f"Dataset has {len(rows)} questions; expected a multiple of "
            f"{QUESTIONS_PER_VIDEO} (QUESTIONS_PER_VIDEO)."
        )
    n = len(rows)
    return [rows[i : i + QUESTIONS_PER_VIDEO] for i in range(0, n, QUESTIONS_PER_VIDEO)]


# ---------------------------------------------------------------------------
# Section 2 — Ingestion helpers
# ---------------------------------------------------------------------------

def delete_index() -> None:
    """Delete the FAISS index directory so the next video starts with a clean slate."""
    shutil.rmtree(str(FAISS_INDEX_PATH), ignore_errors=True)


def ingest_video(video_id: str) -> None:
    """Fetch, chunk, embed, and persist one video's content.

    Args:
        video_id: The 11-character YouTube video ID.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"    Fetching title...", flush=True)
    title = get_video_title(video_id)
    print(f"    Title: {title}", flush=True)

    print(f"    Fetching transcript...", flush=True)
    segments = fetch_transcript(url)
    print(f"    Segments: {len(segments)}", flush=True)

    chunks = chunk_transcript(segments, video_id, title)
    print(f"    Chunks: {len(chunks)}", flush=True)

    vs = create_vector_store(chunks)
    save_index(vs)
    print(f"    Index saved ({vs.index.ntotal} vectors).", flush=True)


# ---------------------------------------------------------------------------
# Section 3 — Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    question: str,
    retriever: Any,
    llm: Any,
) -> tuple[str, list[str]]:
    """Run hybrid retrieval + reranking + Mistral generation for one question.

    Args:
        question: The question string.
        retriever: A ContextualCompressionRetriever wrapping the EnsembleRetriever.
        llm: A ChatOllama (Mistral) instance.

    Returns:
        Tuple of (answer string, list of retrieved chunk page_content strings).
    """
    retrieved = retriever.invoke(question)
    contexts: list[str] = [doc.page_content for doc in retrieved]
    context_str, _ = format_with_sources(retrieved)
    answer: str = (rag_prompt | llm | StrOutputParser()).invoke(
        {"context": context_str, "question": question, "chat_history": []}
    )
    return answer, contexts


# ---------------------------------------------------------------------------
# Section 4 — RAGAS evaluation
# ---------------------------------------------------------------------------

def _safe_mean(val: Any) -> float:
    """Return the mean of a metric value, ignoring NaNs and Nones.

    Args:
        val: A float or list of floats returned by a RAGAS metric.

    Returns:
        Float mean, or nan if no valid values exist.
    """
    if isinstance(val, (int, float)):
        v = float(val)
        return v if not math.isnan(v) else float("nan")
    valid = [float(v) for v in val if v is not None and not math.isnan(float(v))]
    return sum(valid) / len(valid) if valid else float("nan")


def run_ragas_for_group(
    samples: list[dict[str, Any]],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> dict[str, float]:
    """Run RAGAS on one video's answerable questions using Gemini Flash.

    Args:
        samples: List of dicts with keys user_input, response, retrieved_contexts.
        evaluator_llm: A LangchainLLMWrapper around ChatGoogleGenerativeAI.
        evaluator_embeddings: A LangchainEmbeddingsWrapper for scoring embedding similarity.

    Returns:
        Dict mapping our internal metric names to float scores:
        faithfulness, response_relevancy, context_precision.
    """
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]
    dataset = EvaluationDataset.from_list(samples)
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=600, max_retries=10, max_wait=60, max_workers=1),
        raise_exceptions=True,
    )

    scores: dict[str, float] = {}
    for metric in metrics:
        raw_key = metric.name
        internal_key = _RAGAS_KEY_MAP.get(raw_key, raw_key)
        scores[internal_key] = _safe_mean(results[raw_key])

    return scores


# ---------------------------------------------------------------------------
# Section 5 — Refusal check
# ---------------------------------------------------------------------------

def check_refusal(answer: str) -> bool:
    """Return True if the answer contains a known refusal phrase.

    Args:
        answer: The generated answer string.

    Returns:
        True if the model refused to answer, False otherwise.
    """
    return any(phrase in answer.lower() for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate generation, RAGAS evaluation, and report printing/saving."""

    # -----------------------------------------------------------------------
    # 0. Validate required credentials
    # -----------------------------------------------------------------------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "This is required for Gemini Flash evaluation."
        )

    # -----------------------------------------------------------------------
    # 1. Load questions
    # -----------------------------------------------------------------------
    print("Loading questions from eval_dataset.json...", flush=True)
    groups = load_questions(DATASET_PATH)
    n_groups = len(groups)
    total_questions = sum(len(g) for g in groups)

    # Use module-level VIDEO_LABELS if the count matches; otherwise fall back
    # to the video_id of the first answerable row in each group.
    if len(VIDEO_LABELS) == n_groups:
        labels = list(VIDEO_LABELS)
    else:
        labels = [
            next(
                (r["expected_video_id"] for r in g if r.get("expected_video_id")),
                f"Video {i}",
            )
            for i, g in enumerate(groups, start=1)
        ]

    print(f"Loaded {total_questions} questions across {n_groups} videos.\n")

    # -----------------------------------------------------------------------
    # 2. Answer generation (with resume guard)
    # -----------------------------------------------------------------------
    all_rows: list[dict[str, Any]] = []

    if ANSWERS_PATH.exists():
        with ANSWERS_PATH.open() as f:
            cached: list[dict[str, Any]] = json.load(f)
        if len(cached) == total_questions:
            print(
                f"Resume: loaded {total_questions} cached answers from {ANSWERS_PATH}\n",
                flush=True,
            )
            all_rows = cached

    if not all_rows:
        llm = _get_llm()

        for group_idx, group in enumerate(groups, start=1):
            label = labels[group_idx - 1]

            # Derive video_id from the first answerable row in this group.
            video_id: str = next(
                r["expected_video_id"]
                for r in group
                if r.get("expected_video_id")
            )

            print(f"[Video {group_idx}/{n_groups}] {label}  (id={video_id})", flush=True)

            # Fresh index for this video only.
            print("  Deleting old index...", flush=True)
            delete_index()
            ingest_video(video_id)

            # Build retriever once; reuse across all questions in this group.
            vs = load_index()
            all_docs = list(vs.docstore._dict.values())
            ensemble = build_hybrid_retriever(vs, all_docs)
            retriever = build_reranking_retriever(ensemble)

            q_total = len(group)
            for q_idx, row in enumerate(group, start=1):
                question: str = row["question"]
                is_unanswerable: bool = row.get("ground_truth") == "NOT_IN_CONTEXT"
                print(f"  Q{q_idx:02d}/{q_total}: {question[:70]}...", flush=True)

                answer, contexts = run_pipeline(question, retriever, llm)

                all_rows.append(
                    {
                        "video_idx": group_idx,
                        "video_label": label,
                        "video_id": video_id,
                        "question": question,
                        "answer": answer,
                        "retrieved_contexts": contexts,
                        "ground_truth": row.get("ground_truth", ""),
                        "is_unanswerable": is_unanswerable,
                    }
                )

            print(f"  Done with {label}.\n", flush=True)

        # Save all answers before running RAGAS.
        ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ANSWERS_PATH.open("w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"All {total_questions} answers saved → {ANSWERS_PATH}\n", flush=True)

    # -----------------------------------------------------------------------
    # 3. Set up Gemini Flash evaluator
    # -----------------------------------------------------------------------
    print("Setting up Gemini Flash evaluator...", flush=True)
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    print("Gemini evaluator ready.\n", flush=True)

    # -----------------------------------------------------------------------
    # 4. RAGAS per video (answerable rows only)
    # -----------------------------------------------------------------------
    per_video_scores: dict[str, dict[str, float]] = {}

    for group_idx, label in enumerate(labels, start=1):
        print(f"Running RAGAS for Video {group_idx} — {label}...", flush=True)

        ragas_samples = [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
            for r in all_rows
            if r["video_idx"] == group_idx and not r["is_unanswerable"]
        ]

        scores = run_ragas_for_group(ragas_samples, evaluator_llm, evaluator_embeddings)
        per_video_scores[label] = scores
        print(f"  {label}: {scores}\n", flush=True)

    # -----------------------------------------------------------------------
    # 5. Refusal accuracy (unanswerable rows only)
    # -----------------------------------------------------------------------
    unanswerable = [r for r in all_rows if r["is_unanswerable"]]
    refusal_correct = sum(check_refusal(r["answer"]) for r in unanswerable)
    refusal_total = len(unanswerable)

    # -----------------------------------------------------------------------
    # 6. Averages across all 3 videos
    # -----------------------------------------------------------------------
    metric_keys = ["faithfulness", "response_relevancy", "context_precision"]
    averages: dict[str, float] = {}
    for key in metric_keys:
        vals = [
            per_video_scores[lbl][key]
            for lbl in labels
            if not math.isnan(per_video_scores[lbl].get(key, float("nan")))
        ]
        averages[key] = sum(vals) / len(vals) if vals else float("nan")

    # -----------------------------------------------------------------------
    # 7. Print report
    # -----------------------------------------------------------------------
    w = 62
    print("\n" + "=" * w)
    print("PER-VIDEO SCORES  (answerable questions, Gemini Flash judge)")
    print("-" * w)
    for label in labels:
        s = per_video_scores[label]
        print(f"\n  {label}")
        print(f"    {'Faithfulness':<32} {s['faithfulness']:>6.4f}")
        print(f"    {'Response Relevancy':<32} {s['response_relevancy']:>6.4f}")
        print(f"    {'Context Precision':<32} {s['context_precision']:>6.4f}")

    print("\n" + "-" * w)
    print(f"AVERAGE ACROSS ALL {n_groups} VIDEOS")
    print(f"    {'Faithfulness':<32} {averages['faithfulness']:>6.4f}")
    print(f"    {'Response Relevancy':<32} {averages['response_relevancy']:>6.4f}")
    print(f"    {'Context Precision':<32} {averages['context_precision']:>6.4f}")

    print("\n" + "-" * w)
    print(f"REFUSAL ACCURACY                   {refusal_correct}/{refusal_total}")
    print("=" * w)

    # -----------------------------------------------------------------------
    # 8. Save results
    # -----------------------------------------------------------------------
    result: dict[str, Any] = {
        "per_video": per_video_scores,
        "average": averages,
        "refusal_accuracy": {
            "correct": refusal_correct,
            "total": refusal_total,
            "score": round(refusal_correct / refusal_total, 4) if refusal_total else 0.0,
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
