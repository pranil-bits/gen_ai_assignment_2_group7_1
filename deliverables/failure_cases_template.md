# Failure-Case Analysis

## Recorded examples

1. **Warehouse architecture missed the exact document (T01, text)**
   - The top results included a multicluster warehouse document but ranked it outside the relevant source set.
   - Likely causes: broad architecture wording and dense similarity dominated by shared "warehouse" terms.
   - Fix: metadata filter for `topic=performance`, a warehouse-size synonym ontology, and BM25 or phrase boost.

2. **Credit unit question did not retrieve the credit table (TAB02, table)**
   - The gold source `024_CreditConsumptionTable.pdf` was not in the top five, while FinOps text chunks were retrieved.
   - Likely causes: modality not promoted in the query and table summary omitted enough numeric context.
   - Fix: modality classifier, table-summary expansion, and metadata filter `modality=table`.

3. **Data-sharing comparison table was missed (TAB10, table)**
   - Only unrelated semantic-layer tables were returned.
   - Likely causes: the abbreviated question lacked provider/consumer/product terms and no table-aware routing was used.
   - Fix: metadata ontology expansion for `topic=data_sharing`, query expansion, and table-only reranking.

## Required evidence
Record at least three actual failures from `evaluation_per_case.csv`, including:

1. **Incorrect or missing retrieval**
   - Query
   - Retrieved source and rank
   - Expected source and page
   - Likely cause
   - Proposed fix

2. **Modalilty fusion failure**
   - Table or image not retrieved
   - Dense-text-only comparison
   - Proposed metadata or OCR fix

3. **Answer grounding failure**
   - Unsupported or partially supported context
   - Guardrail or citation failure
   - Proposed prompt or ranking fix
