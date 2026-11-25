# Guardrail Specification

## Input checks
- Reject empty or trivial queries.
- Reject queries longer than 1000 characters.
- Reject prompt-injection phrases.
- Reject direct requests for passwords, identifiers, social-security numbers, Aadhaar numbers, or credit-card details.
- Retrieve only public, allow-listed ontology metadata.

## Output checks
- Return a concise answer.
- Cite at least one document and page for factual claims.
- Refuse with `Insufficient evidence` when retrieval evidence is unsupported.
- Do not expose hidden chain-of-thought.
- Strip credentials, PII, secrets, and unverified personal attributes.

## Ontology routing
The metadata ontology routes on `topic`, `tool`, `doc_type`, `modality`, `year`, `source_url`, and `sensitivity`.
Only `sensitivity=public` records are searchable. Table and image records are filtered by `modality`.
