# Multimodal Document Retrieval - Part A

This repository contains the implementation and supporting materials for PCAM ZG516 Assignment 2 Part A.

## Run the project from scratch

These steps assume you have cloned the repository, have the source documents in `corpus/`, and are running commands from the repository root.

### 1. Set up Python

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Check the corpus

The corpus should contain the supported PDF and PowerPoint documents under `corpus/` (for example `corpus/pdf/`, `corpus/web/`, and `corpus/ppt/`). The inventory command reports how many documents, pages/slides, tables, and images were found:

```powershell
python code/corpus_inventory.py
```

It writes `deliverables/corpus_inventory.csv` and `deliverables/corpus_inventory_summary.json`.

### 3. Build the searchable index

The default script ingests a small deterministic pilot set. To index all supported PDF/PPTX documents, set `PART_A_INGEST_ALL=1`. OCR is disabled in this example because OCR over the full corpus is slow; image-derived text will therefore not be indexed.

Windows PowerShell:

```powershell
$env:PART_A_INGEST_ALL = "1"
$env:PART_A_SKIP_OCR = "1"
python code/pilot_ingestion.py
```

Linux or macOS:

```bash
PART_A_INGEST_ALL=1 PART_A_SKIP_OCR=1 python code/pilot_ingestion.py
```

The script creates the Chroma index and ingestion outputs in `corpus/pilot_out/`. **Each ingestion run replaces the existing `pilot` collection.** Do not run ingestion if you need to preserve an existing index. To try the smaller pilot set instead, leave `PART_A_INGEST_ALL` unset. To run OCR on the selected images, leave `PART_A_SKIP_OCR` unset; processing will take longer.

### 4. Ask questions

Use the same interactive demo with different modes:

```powershell
# Retrieve and display passages only; does not load an answer-generating LLM
python code/demo_part_a.py --mode retrieval

# Retrieve passages and generate an answer from them with the default Qwen model
python code/demo_part_a.py --mode generate

# Show both retrieved passages and the generated answer
python code/demo_part_a.py --mode both --prompt-variant zero_shot
```

Type a question at the prompt; enter `exit` to quit. The embedding model (BGE) finds passages. In `generate` and `both` modes, the open-weight Qwen model uses those passages to draft an answer with source references.

The first generation run downloads the LLM from Hugging Face and requires internet access and disk space. It runs locally on CPU by default and may be slow. Choose another compatible causal language model with `--llm-model MODEL_ID_OR_LOCAL_PATH`. Select one of the implemented prompt templates with `--prompt-variant` (`zero_shot`, `f_string`, `three_shot`, `jinja2`, or `react_or_create`). Retrieval-only mode does not load the LLM.

### 5. Run retrieval and prompt evaluations

Retrieval evaluation (30-question golden set):

```powershell
python code/part_a_evaluate.py --k 5 --repeat 5
```

Prompt comparison runs actual generations for each prompt variant using the same model and golden questions. Start with a short smoke test:

```powershell
python code/prompt_ablation.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 3
```

Run the full golden set after the smoke test:

```powershell
python code/prompt_ablation.py --model Qwen/Qwen2.5-0.5B-Instruct
```

This can take a long time on CPU because it generates answers for every question and prompt variant. The automatic keyword scores are proxies, not expert or LLM-judge scores; review the responses and metrics before making quality claims.

## What should be committed to Git?

Commit source code, setup instructions, and **curated assignment materials** that your partner needs to understand, review, and reproduce the work. This normally includes:

- `code/`, `requirements.txt`, `.gitignore`, and this README.
- `deliverables/golden_set.jsonl`, `design.md`, `guardrail_spec.md`, and the report/failure-analysis templates.
- Final, reviewed evaluation results and the completed report when ready to submit. Clearly label preliminary or partial results; do not present smoke-test output as a final benchmark.
- A contribution statement and presentation materials when your group has prepared them.

The `deliverables/` directory is not ignored, so its files can be committed. Include generated inventory/evaluation outputs when they are useful as reviewed evidence or required for the assignment; otherwise they can be regenerated using the commands above. Keep the golden set and other hand-curated materials in Git because they are inputs to evaluation, not disposable build outputs.

Do **not** commit `.venv/`, downloaded model caches, or the generated Chroma database and ingestion outputs under `corpus/pilot_out/`; those can be rebuilt and may be large.

The source documents under `corpus/` are required to rebuild the index, but only add them to a shared remote if your group is allowed to redistribute them and the repository can accommodate their size. Publicly accessible documents are not automatically redistributable. If you do not commit them, tell your partner where to obtain the permitted corpus and how to place it under `corpus/`.

## Current implementation notes

- The complete index can search extracted document text and tables.
- Full-corpus OCR is disabled in the documented command above to keep ingestion practical. Image count in the inventory does not mean image contents are searchable.
- The prompt comparison uses an automatic keyword-based proxy. Human review and final assignment evaluation are still needed.
- Review the golden-set questions and expected answers as a group before reporting results.
- Do not add confidential, proprietary, or personal documents.
- AI-assisted code generation was used for boilerplate and project structure. The group is responsible for verifying claims, results, report analysis, and contribution statements.
