"""Local Ollama-judged RAGAS evaluation of the RAG pipeline across videos.

Run as:
    python -m src.evaluation.local_eval

Flow:
    1. Load evaluation/eval_dataset.json (questions grouped per video).
    2. For each video: delete index → re-ingest → run full pipeline for all Qs.
    3. Save all answers to evaluation/results/gemini_answers.json
       (reused as-is if already cached — only the judge changes, not the answers).
    4. Evaluate each answerable question individually with RAGAS
       (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference)
       using llama3.1:8b via local Ollama as the judge.
    5. Checkpoint each question's scores to evaluation/results/local_eval_scores.json
       immediately after evaluation — safe to interrupt and resume at any point.
    6. String-match refusal phrases on unanswerable questions.
    7. Print per-video + average scores and refusal accuracy.
    8. Save final aggregated results to evaluation/results/local_eval_results.json.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, OLLAMA_BASE_URL
from src.generation.chains import _get_llm
from src.generation.citations import _REFUSAL_PHRASES, format_with_sources
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
# Reuse cached answers from gemini_eval — only the judge changes, not the answers.
ANSWERS_PATH = Path("evaluation/results/gemini_answers.json")
# Per-question score checkpoint — written after every question, safe to resume.
SCORES_PATH = Path("evaluation/results/local_eval_scores.json")
RESULTS_PATH = Path("evaluation/results/local_eval_results.json")

QUESTIONS_PER_VIDEO = 10

# Human-readable labels in file order
VIDEO_LABELS = ["Vidya Balan", "Kriti Sanon", "Shahid Kapoor"]

# Normalize RAGAS result keys → our internal names (handles older/newer RAGAS versions)
_RAGAS_KEY_MAP: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "response_relevancy",
    "response_relevancy": "response_relevancy",
    "llm_context_precision_without_reference": "context_precision",
}

_METRIC_KEYS = ["faithfulness", "response_relevancy", "context_precision"]


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
    print("    Fetching title...", flush=True)
    title = get_video_title(video_id)
    print(f"    Title: {title}", flush=True)

    print("    Fetching transcript...", flush=True)
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
# Section 4 — RAGAS evaluation (per-question)
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


def evaluate_one(
    sample: dict[str, Any],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> dict[str, float]:
    """Run RAGAS on a single question and return its metric scores.

    Evaluates all three metrics in one call:
        - Faithfulness
        - ResponseRelevancy
        - LLMContextPrecisionWithoutReference

    With ``raise_exceptions=False``, any metric that the judge LLM fails to
    score is returned as ``float("nan")`` instead of raising — the caller
    logs NaN values and continues to the next question.

    Args:
        sample: Dict with keys ``user_input``, ``response``, and
            ``retrieved_contexts`` — the format expected by RAGAS.
        evaluator_llm: A :class:`LangchainLLMWrapper` around the local
            ChatOllama judge model.
        evaluator_embeddings: A :class:`LangchainEmbeddingsWrapper` used by
            ResponseRelevancy to embed the question and answer.

    Returns:
        Dict mapping internal metric names to float scores.
        Keys: ``faithfulness``, ``response_relevancy``, ``context_precision``.
        Failed metrics are ``float("nan")``.
    """
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]
    dataset = EvaluationDataset.from_list([sample])
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=1200, max_workers=1),
        raise_exceptions=False,
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
    """Orchestrate generation, per-question RAGAS evaluation, and reporting."""

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
    # 2. Answer generation (reuse cached answers if available)
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

            video_id: str = next(
                r["expected_video_id"]
                for r in group
                if r.get("expected_video_id")
            )

            print(f"[Video {group_idx}/{n_groups}] {label}  (id={video_id})", flush=True)

            print("  Deleting old index...", flush=True)
            delete_index()
            ingest_video(video_id)

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

        ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ANSWERS_PATH.open("w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"All {total_questions} answers saved → {ANSWERS_PATH}\n", flush=True)

    # -----------------------------------------------------------------------
    # 3. Set up local Ollama evaluator (qwen3-nothink)
    # -----------------------------------------------------------------------
    print(
        f"Setting up local Ollama evaluator (qwen3-nothink at {OLLAMA_BASE_URL})...",
        flush=True,
    )
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model="qwen3-nothink",
            base_url=OLLAMA_BASE_URL,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    print("Evaluator ready.\n", flush=True)

    # -----------------------------------------------------------------------
    # 4. Per-question RAGAS evaluation with checkpoint
    # -----------------------------------------------------------------------
    # Load existing checkpoint so interrupted runs resume from where they left off.
    # Keys are question strings; values are {metric: score} dicts.
    scores_checkpoint: dict[str, dict[str, float]] = {}
    if SCORES_PATH.exists():
        with SCORES_PATH.open() as f:
            scores_checkpoint = json.load(f)

    answerable_rows = [r for r in all_rows if not r["is_unanswerable"]]
    already_scored = sum(
        1 for r in answerable_rows
        if f"{r['video_idx']}:{r['question']}" in scores_checkpoint
    )

    if already_scored:
        print(
            f"Resuming: {already_scored}/{len(answerable_rows)} questions already scored.\n",
            flush=True,
        )

    for r in answerable_rows:
        question = r["question"]
        ck_key = f"{r['video_idx']}:{question}"

        if ck_key in scores_checkpoint:
            print(f"  [cached] {question[:70]}...", flush=True)
            continue

        label = r["video_label"]
        print(f"  [{label}] {question[:70]}...", flush=True)

        sample = {
            "user_input": question,
            "response": r["answer"],
            "retrieved_contexts": r["retrieved_contexts"],
        }

        scores = evaluate_one(sample, evaluator_llm, evaluator_embeddings)

        # Log any NaN metrics so failures are visible without stopping the run.
        nan_metrics = [k for k, v in scores.items() if math.isnan(v)]
        if nan_metrics:
            print(f"    [WARN] NaN for: {', '.join(nan_metrics)}", flush=True)

        scores_checkpoint[ck_key] = scores

        # Write the checkpoint after every question — safe to interrupt.
        SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SCORES_PATH.open("w") as f:
            json.dump(scores_checkpoint, f, indent=2)

    print(f"\nAll {len(answerable_rows)} answerable questions scored.\n", flush=True)

    # -----------------------------------------------------------------------
    # 5. Aggregate per-video scores from individual question scores
    # -----------------------------------------------------------------------
    per_video_scores: dict[str, dict[str, float]] = {}

    for group_idx, label in enumerate(labels, start=1):
        group_rows = [
            r for r in all_rows
            if r["video_idx"] == group_idx and not r["is_unanswerable"]
        ]
        per_video_scores[label] = {
            metric: _safe_mean([
                scores_checkpoint[f"{r['video_idx']}:{r['question']}"][metric]
                for r in group_rows
                if f"{r['video_idx']}:{r['question']}" in scores_checkpoint
            ])
            for metric in _METRIC_KEYS
        }

    # -----------------------------------------------------------------------
    # 6. Refusal accuracy (unanswerable rows only)
    # -----------------------------------------------------------------------
    unanswerable = [r for r in all_rows if r["is_unanswerable"]]
    refusal_correct = sum(check_refusal(r["answer"]) for r in unanswerable)
    refusal_total = len(unanswerable)

    # -----------------------------------------------------------------------
    # 7. Averages across all videos
    # -----------------------------------------------------------------------
    averages: dict[str, float] = {}
    for key in _METRIC_KEYS:
        vals = [
            per_video_scores[lbl][key]
            for lbl in labels
            if not math.isnan(per_video_scores[lbl].get(key, float("nan")))
        ]
        averages[key] = sum(vals) / len(vals) if vals else float("nan")

    # -----------------------------------------------------------------------
    # 8. Print report
    # -----------------------------------------------------------------------
    w = 62
    print("\n" + "=" * w)
    print("PER-VIDEO SCORES  (answerable questions, qwen3-nothink local judge)")
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
    # 9. Save results
    # -----------------------------------------------------------------------
    result: dict[str, Any] = {
        "judge": f"qwen3-nothink @ {OLLAMA_BASE_URL}",
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
