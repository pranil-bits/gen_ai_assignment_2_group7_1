import math


MODALITY_TERMS = {
    "table": (
        "table", "column", "row", "credit consumption", "compare", "comparison",
        "edition", "columns", "warehouse size", "units",
    ),
    "image": (
        "diagram", "architecture", "flow", "components", "shown", "image",
        "chart", "graph", "illustration", "pipeline", "visual",
    ),
}


def classify_modality(question, default="text"):
    lowered = question.lower()
    scores = {
        modality: sum(lowered.count(term) for term in terms)
        for modality, terms in MODALITY_TERMS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default


def percentile(values, percentile_value):
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def format_collection_error(collection_name, root):
    return (
        f"Chroma collection '{collection_name}' was not found at {root}. "
        "Run 'python code/pilot_ingestion.py' or set PART_A_INGEST_ALL=1 "
        "for the full corpus before evaluating."
    )


def reject_query(query):
    lowered = query.lower()
    if len(query) > 1000:
        return True, "query_too_long"
    injection_terms = ("ignore previous", "system prompt", "developer message", "reveal secrets")
    if any(term in lowered for term in injection_terms):
        return True, "prompt_injection"
    personal_terms = (
        "my password", "password", "social security", "credit card",
        "aadhaar", "personal identifier", "employee id",
    )
    if any(term in lowered for term in personal_terms):
        return True, "personal_data"
    if len(query.strip()) < 3:
        return True, "empty_or_trivial"
    return False, ""


def chunk_text(text, size=250, overlap=40):
    import re

    normalized = re.sub(r"\s+", " ", text).strip()
    words = normalized.split()
    if not words:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if len(chunk.split()) >= 12:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def add_source_citations(answer, evidence):
    cleaned = answer.strip()
    if not cleaned:
        return "Insufficient evidence"
    if cleaned.lower().startswith("insufficient evidence"):
        return cleaned
    citations = "; ".join(
        f"[{item['doc_id']}, page {item['page']}]" for item in evidence
    )
    if not citations:
        return f"{cleaned}\n\nInsufficient evidence"
    return f"{cleaned}\n\nSources: {citations}"
