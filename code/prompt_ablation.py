import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from demo_part_a import (
    DEFAULT_LLM_MODEL,
    add_source_citations,
    encode_prompt,
    generate_answer,
    load_llm,
    search,
)
from prompts import VARIANTS, render

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "deliverables" / "golden_set.jsonl"
OUT = ROOT / "deliverables"
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "which", "with",
}


def keywords(text):
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


def load_golden():
    with open(GOLDEN, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(model_name=DEFAULT_LLM_MODEL, limit=None, top_k=5):
    cases = load_golden()
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("The golden set has no cases to evaluate.")

    tokenizer, llm = load_llm(model_name)
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path=str(ROOT / "corpus" / "pilot_out" / "chroma"))
    collection = client.get_collection("pilot")
    results = defaultdict(list)

    for case_number, case in enumerate(cases, 1):
        retrieved = search(case["question"], embedding_model, collection, k=top_k)
        evidence = [
            {
                "doc_id": item["doc_id"],
                "page": item["page"],
                "modality": item["modality"],
                "text": text,
            }
            for text, item in zip(retrieved["documents"][0], retrieved["metadatas"][0])
        ]
        context_words = keywords(" ".join(item["text"] for item in evidence))
        expected_words = keywords(case["answer"])

        for variant in VARIANTS:
            prompt = render(
                variant.name,
                case["question"],
                evidence,
                modality=case["modality"],
            )
            input_token_count = encode_prompt(prompt, tokenizer).shape[-1]
            started = time.perf_counter()
            answer = add_source_citations(
                generate_answer(prompt, tokenizer, llm),
                evidence,
            )
            latency = time.perf_counter() - started
            answer_word_set = keywords(answer)
            generated_token_count = len(tokenizer(answer, add_special_tokens=False).input_ids)

            results[variant.name].append({
                "question_id": case["id"],
                "answer_keyword_recall": (
                    len(expected_words & answer_word_set) / len(expected_words)
                    if expected_words else 0.0
                ),
                "context_keyword_precision": (
                    len(answer_word_set & context_words) / len(answer_word_set)
                    if answer_word_set else 0.0
                ),
                "input_tokens": input_token_count,
                "output_tokens": generated_token_count,
                "latency_seconds": latency,
                "answer": answer,
            })
        print(f"Evaluated {case_number}/{len(cases)} questions")

    summary = []
    for variant in VARIANTS:
        measurements = results[variant.name]
        latencies = sorted(item["latency_seconds"] for item in measurements)
        p95_index = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)) - 1))
        summary.append({
            "prompt_variant": variant.name,
            "model": model_name,
            "questions": len(measurements),
            "answer_keyword_recall": round(
                sum(item["answer_keyword_recall"] for item in measurements) / len(measurements), 4
            ),
            "context_keyword_precision": round(
                sum(item["context_keyword_precision"] for item in measurements) / len(measurements), 4
            ),
            "average_input_tokens": round(
                sum(item["input_tokens"] for item in measurements) / len(measurements), 1
            ),
            "average_output_tokens": round(
                sum(item["output_tokens"] for item in measurements) / len(measurements), 1
            ),
            "p50_latency_seconds": round(latencies[len(latencies) // 2], 4),
            "p95_latency_seconds": round(latencies[p95_index], 4),
            "generation": "greedy (do_sample=False)",
        })

    with open(OUT / "prompt_ablation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved real LLM prompt comparison to {OUT / 'prompt_ablation.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare prompt variants with the same LLM and golden questions.")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (useful for a quick smoke test).")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(model_name=args.model, limit=args.limit, top_k=args.top_k)
