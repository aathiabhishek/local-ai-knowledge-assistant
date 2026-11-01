import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from rouge_score import rouge_scorer
import re


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import source_retrieval_metrics
from src.config import EMBEDDING_MODEL, MODEL_PATH, TOP_K
from src.retriever import HybridRetriever
from src.llm import LocalLLM
from src.ingestion import load_uploaded_files

DOCS_DIR = ROOT / "evaluation" / "benchmark" / "documents"

REFUSAL_PHRASES = (
    "couldn't find",
    "could not find",
    "can't find",
    "cannot find",
    "cannot answer",
    "can't answer",
    "don't know",
    "do not know",
    "not enough information",
    "not mentioned",
    "not found",
    "no information",
    "not in the",
    "not provided",
    "not specified",
    "not available",
    "does not contain",
)


_CITATION_PATTERN = re.compile(r"\[[^\]]+\]")


def strip_citations(text: str) -> str:
    """Remove [source.md] brackets before scoring ROUGE/Semantic Similarity."""
    return _CITATION_PATTERN.sub("", text).strip()

class LocalFile:
    """Adapter that mimics Streamlit's UploadedFile (name + getvalue)."""

    def __init__(self, path: Path):
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._bytes


def load_documents(folder: Path) -> List[Dict]:
    if not folder.exists():
        return []
    files = [
        LocalFile(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
    ]
    warnings = []
    docs = load_uploaded_files(files, warnings)
    for w in warnings:
        print(f"Warning: {w}")
    return docs


def is_unanswerable(expected: str) -> bool:
    return expected.lower().startswith("not in the document")


def looks_like_refusal(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return any(p in text for p in REFUSAL_PHRASES)


def compute_semantic_similarity(embedding_model, text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vecs = embedding_model.encode(
        [text_a, text_b],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vec_a, vec_b = vecs
    return float(np.dot(vec_a, vec_b))


def evaluate_lexical_overlap(scorer, reference: str, hits: List[Dict]) -> float:
    """Diagnostic answer-token coverage; not source retrieval correctness."""
    context = " ".join(h.get("text", "") for h in hits)
    return scorer.score(reference, context)["rouge1"].recall if hits else 0.0


def run_evaluation(data_path: Path, output_path: Path):
    if not data_path.exists():
        raise SystemExit(f"Question file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if not eval_data:
        print("No questions found.")
        return

    for entry in eval_data:
        reference = entry.get("expected") or entry.get("reference", "")
        if isinstance(reference, list):
            reference = next(iter(reference), "")
        if not is_unanswerable(reference) and not entry.get("relevant_sources"):
            raise SystemExit(
                "Benchmark lacks relevant_sources. Rebuild it with "
                "evaluation/build_large_benchmark.py before scoring source retrieval."
            )

    documents = load_documents(DOCS_DIR)
    if not documents:
        raise SystemExit(f"No documents found in {DOCS_DIR}. Check DOCS_DIR.")
    print(f"Loaded {len(documents)} chunks from {DOCS_DIR}")

    retriever = HybridRetriever(EMBEDDING_MODEL)
    retriever.build(documents)
    llm = LocalLLM(MODEL_PATH)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []

    print(f"Starting evaluation on {len(eval_data)} test cases...\n")

    for idx, entry in enumerate(eval_data, start=1):
        question = entry["question"]
        reference = entry.get("expected") or entry["reference"]
        if isinstance(reference, list):
            reference = next(iter(reference), "")

        t0 = time.perf_counter()
        hits = retriever.search(question, top_k=TOP_K)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        generated = llm.generate_rag_answer(
            question=question,
            retrieved_chunks=hits,
            history=[],
            temperature=0.0,
        )
        generation_ms = (time.perf_counter() - t1) * 1000.0

        if is_unanswerable(reference):
            kind = "unanswerable"
            passed = looks_like_refusal(generated)
            r1 = rl = ans_recall = sem_sim = ctx_recall = hit_k = rr = None
        else:
            kind = "answerable"
            passed = None
            
            clean_generated = strip_citations(generated)

            s = scorer.score(reference, clean_generated)
            r1 = s["rouge1"].fmeasure
            rl = s["rougeL"].fmeasure
            ans_recall = s["rouge1"].recall
            sem_sim = compute_semantic_similarity(
                retriever.embedding_model, reference, generated
            )

            ctx_recall = evaluate_lexical_overlap(scorer, reference, hits)
            source_metrics = source_retrieval_metrics(entry["relevant_sources"], hits)
            rr = source_metrics["source_reciprocal_rank"]
            hit_k = source_metrics["source_hit_at_k"]

        results.append({
            "question": question,
            "relevant_sources": entry.get("relevant_sources", []),
            "type": kind,
            "generated": generated,
            "reference": reference,
            "context_rouge1_recall": ctx_recall,
            "source_hit_at_k": hit_k,
            "source_reciprocal_rank": rr,
            "answer_rouge1_recall": ans_recall,
            "rouge1_f1": r1,
            "rougeL_f1": rl,
            "semantic_similarity": sem_sim,
            "declined_correctly": passed,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "retrieved_sources": [
                f'{h.get("source")} p{h.get("page")}' for h in hits
            ],
        })

        print(f"[{idx}/{len(eval_data)}] Q: {question}")
        print(f"Generated: {generated}")
        print(f"Reference: {reference}")
        if kind == "answerable":
            print(
                f"[Retrieval]  Context Recall: {ctx_recall:.1%} | Hit@{TOP_K}: {int(hit_k)} | RR: {rr:.2f} | {retrieval_ms:.1f}ms\n"
                f"[Generation] Answer Recall:  {ans_recall:.1%} | Semantic Sim: {sem_sim:.1%} | ROUGE-1 F1: {r1:.1%} | {generation_ms:.0f}ms"
            )
        else:
            print(f"[Guardrail]  Declined correctly: {passed} ({generation_ms:.0f}ms)")
        print("-" * 60)

    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    print("\n" + "=" * 60)
    print("FINAL RAG BENCHMARK SUMMARY")
    print("=" * 60)
    if answerable:
        n = len(answerable)
        avg = lambda key: sum(r[key] for r in answerable) / n
        print(
            f"RETRIEVAL METRICS (Top-{TOP_K}, {n} answerable queries):\n"
            f"  • Context Recall (ROUGE-1): {avg('context_rouge1_recall'):.1%}\n"
            f"  • Source Hit Rate @ {TOP_K}:             {avg('source_hit_at_k'):.1%}\n"
            f"  • Source-passage MRR:     {avg('source_reciprocal_rank'):.3f}\n"
            f"  • Avg Retrieval Latency:    {avg('retrieval_ms'):.1f} ms\n"
        )
        print(
            f"GENERATION METRICS (Llama 3.1, {n} answerable queries):\n"
            f"  • Semantic Answer Sim:      {avg('semantic_similarity'):.1%}\n"
            f"  • Answer Recall (ROUGE-1):  {avg('answer_rouge1_recall'):.1%}\n"
            f"  • Answer ROUGE-1 F1:        {avg('rouge1_f1'):.1%}\n"
            f"  • Answer ROUGE-L F1:        {avg('rougeL_f1'):.1%}\n"
            f"  • Avg Generation Latency:   {avg('generation_ms'):.0f} ms"
        )
    if unanswerable:
        ok = sum(1 for r in unanswerable if r["declined_correctly"])
        total_unans = len(unanswerable)
        print(
            f"\nOUT-OF-SCOPE GUARDRAIL ({total_unans} unanswerable queries):\n"
            f"  • Negative Rejection Rate:  {ok}/{total_unans} ({ok/total_unans:.0%})"
        )
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed JSON report to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "evaluation" / "benchmark" / "questions.json"))
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results_source_grounded.json"))
    args = parser.parse_args()
    run_evaluation(Path(args.data), Path(args.out))
