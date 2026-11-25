from dataclasses import dataclass

from jinja2 import Template


@dataclass(frozen=True)
class PromptVariant:
    name: str
    template: str


VARIANTS = [
    PromptVariant(
        "zero_shot",
        """Answer the question using only the retrieved evidence. Cite the document and page.
Question: {{ question }}
Evidence:
{{ evidence }}
""",
    ),
    PromptVariant(
        "f_string",
        """Use the evidence below to answer: {question}
Rules: be concise, grounded, and cite source document/page.
Evidence: {evidence}
""",
    ),
    PromptVariant(
        "three_shot",
        """Examples:
1. If evidence says storage and compute are separate, cite the warehouse architecture page.
2. If the evidence is a table, include the table context and source.
3. If evidence does not support an answer, say "Insufficient evidence".

Question: {{ question }}
Evidence:
{{ evidence }}
Return the answer and citations.
""",
    ),
    PromptVariant(
        "jinja2",
        """{% if evidence %}
Ontology scope: {{ modality }}
Question: {{ question }}
Evidence:
{% for item in evidence %}
- [{{ item.doc_id }}, page {{ item.page }}] {{ item.text }}
{% endfor %}
Answer only from the evidence. Include a source for every factual claim.
{% else %}
Return "Insufficient evidence" and explain that no supporting source was retrieved.
{% endif %}
""",
    ),
    PromptVariant(
        "react_or_create",
        """Reason in stages without exposing private chain-of-thought:
1. Check whether evidence supports the question.
2. Select the relevant modality and source.
3. Compose a concise grounded answer.
4. Cite document and page.
5. Reject unsupported claims.

Question: {{ question }}
Evidence:
{{ evidence }}
""",
    ),
]


def render(variant_name, question, evidence, modality="text"):
    variant = next(item for item in VARIANTS if item.name == variant_name)
    evidence_items = evidence if isinstance(evidence, list) else []
    if evidence_items and isinstance(evidence_items[0], dict):
        evidence_text = "\n".join(
            f"[{item['doc_id']}, page {item['page']}, {item['modality']}] {item['text']}"
            for item in evidence_items
        )
    elif evidence_items and hasattr(evidence_items[0], "doc_id"):
        evidence_text = "\n".join(
            f"[{item.doc_id}, page {item.page}] {item.text}" for item in evidence_items
        )
    else:
        evidence_text = evidence

    if variant_name == "f_string":
        return variant.template.format(question=question, evidence=evidence_text)

    context = evidence_items if variant_name == "jinja2" and evidence_items else evidence_text
    return Template(variant.template).render(
        question=question,
        evidence=context,
        modality=modality,
    )
