"""Pilot ingestion v2 (script version of pilot_ingestion.ipynb).
Run:  python code/pilot_ingestion.py   (from the assignment2 folder, or from anywhere)
"""

# ======== config ========
import os, re, io, csv, json, time, random
from pathlib import Path
from collections import Counter
INGEST_ALL = os.environ.get("PART_A_INGEST_ALL", "0") == "1"

# Finds the corpus folder whether you run from assignment2 or from assignment2/code
_here = Path(__file__).resolve().parent
CORPUS_ROOT = next((p for p in (Path.cwd() / "corpus", _here / "corpus", _here.parent / "corpus") if p.exists()),
                   Path("corpus"))
print("Using corpus folder:", CORPUS_ROOT.resolve())
PILOT_FILES = []                    # e.g. ["corpus/pdf/003_x.pdf"]; leave empty to auto-pick
PILOT_PER_FOLDER = {"pdf": 2, "web": 2, "ppt": 1}
# Files whose name contains one of these are ALWAYS added (known table-heavy pages, to validate table extraction)
ALSO_INCLUDE = ["CreditConsumption", "intro-editions", "warehouses-overview", "finops-for-data-cloud"]

CHUNK_WORDS, OVERLAP_WORDS = 250, 40   # approx tokens = words x 1.3
MIN_IMG_PX, MIN_IMG_BYTES = 120, 4000  # skip logos and tiny icons
MIN_OCR_CHARS = 20
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

OUT = CORPUS_ROOT / "pilot_out"
OUT.mkdir(parents=True, exist_ok=True)

# ======== helpers ========
# ---- source URLs (from the download report) ----
def load_urls():
    urls, p = {}, CORPUS_ROOT / "report.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("file"):
                urls[os.path.splitext(r["file"])[0]] = r["url"]
    return urls
URLS = load_urls()

# ---- ontology (rule-based first pass; refine later) ----
def guess_tool(url):
    u = url.lower()
    s, z = "snowflake" in u, "zenlytic" in u
    a = any(k in u for k in ("azure", "microsoft", "entra", "fabric"))
    if z: return "zenlytic"
    if s and a: return "cross_stack"
    if s: return "snowflake"
    if a: return "azure"
    return "cross_stack"

def guess_doc_type(url, ext):
    u = url.lower()
    if ext == ".pptx": return "deck"
    if any(k in u for k in ("docs.snowflake.com", "learn.microsoft.com", "docs.getdbt.com", "intercom.help")): return "docs_page"
    if "ebook" in u: return "ebook"
    if "whitepaper" in u or "wp-content" in u: return "whitepaper"
    if "blog" in u: return "blog"
    return "guide"

TOPIC_RULES = {
    "cost": ["credit", "pricing", "cost", "finops", "billing", "spend"],
    "security_identity": ["security", "entra", "sso", "rbac", "masking", "encryption", "authentication", "access control"],
    "governance": ["governance", "purview", "horizon", "lineage", "catalog", "compliance"],
    "ingestion": ["snowpipe", "ingest", "data factory", "loading", "copy into", "streaming", "kafka"],
    "architecture": ["architecture", "landing zone", "lakehouse", "medallion", "reference"],
    "semantic_layer": ["semantic layer", "metricflow", "semantic view", "metrics layer", "zenlytic"],
    "ai_ml": ["cortex", "llm", "machine learning", "ai agent", "snowpark ml", "genai"],
    "performance": ["warehouse", "performance", "query", "caching", "scaling", "dynamic table"],
    "data_sharing": ["data sharing", "marketplace", "native app", "listing", "clean room"],
    "industry_use_case": ["manufacturing", "retail", "healthcare", "financial services", "telecom", "supply chain", "industry"],
}
def guess_topic(text, label=""):
    # label = title + url: counted 5x so the document's own name outweighs stray words in the body
    t = text.lower()[:6000]
    lab = label.lower()
    scores = {k: sum(t.count(w) + 5 * lab.count(w) for w in ws) for k, ws in TOPIC_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

def guess_year(url, text):
    m = re.search(r"(20[12]\d)", url)
    if m and int(m.group(1)) <= 2026:
        return int(m.group(1))
    ys = [int(y) for y in re.findall(r"\b(20[12]\d)\b", text[:3000]) if 2015 <= int(y) <= 2026]
    return Counter(ys).most_common(1)[0][0] if ys else 0   # 0 = unknown (Chroma cannot hold None)

def pretty_title(stem):
    """Readable title from the file name (page text gave poor titles such as 'Guides' or 'Snowflake')."""
    s = re.sub(r"^\d+_", "", stem)
    s = re.sub(r"^[0-9a-f]{16,}_", "", s)
    s = s.replace("_20", " ").replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()

# ---- text chunking (page-level so every chunk has a page for citation) ----
def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    units = []
    for p in (p for p in paras if p):
        w = p.split()
        if len(w) <= size:
            units.append(p)
        else:
            units += [" ".join(w[k:k + size]) for k in range(0, len(w), size)]
    chunks, cur, n = [], [], 0
    for u in units:
        c = len(u.split())
        if cur and n + c > size:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(u); n += c
    if cur:
        chunks.append(" ".join(cur))
    if overlap and len(chunks) > 1:
        chunks = [chunks[0]] + [" ".join(chunks[j - 1].split()[-overlap:]) + " " + chunks[j]
                                for j in range(1, len(chunks))]
    return [c for c in chunks if len(c.split()) >= 12]

# ---- tables: markdown copy (stored) + short summary (searched) ----
def clean_rows(rows):
    rows = [[("" if c is None else re.sub(r"\s+", " ", str(c)).strip()) for c in r] for r in rows]
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]

def table_markdown(rows):
    w = len(rows[0])
    md_ = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * w + "\n"
    return md_ + "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])

def table_summary(rows, page):
    header = [c for c in rows[0] if c]
    first_col = [r[0] for r in rows[1:6] if r and r[0]]
    return (f"Table on page {page} with {len(rows) - 1} rows. Columns: {', '.join(header)[:200]}. "
            f"Rows include: {'; '.join(first_col)[:200]}.")

# ---- table quality filter: drops website footers/menus that look like tables ----
YES_NO = {"yes", "no", "y", "n", "true", "false", "x", "n/a", "na", "supported", "not supported", "-", "\u2713", "\u2717"}

def table_ok(rows, bbox=None, page_h=None):
    """Return (keep, reason). Heuristic filter; dropped tables are saved for review."""
    cells = [c for r in rows for c in r]
    if len(rows) < 2 or len(rows[0]) < 2:
        return False, "too small"
    if sum(bool(c) for c in cells) / len(cells) < 0.6:
        return False, "mostly empty"
    if sum(bool(c) for c in rows[0]) < 2:
        return False, "header has <2 filled cells"
    if bbox is not None and page_h and bbox[1] > 0.85 * page_h:
        return False, "starts in bottom 15% of page (footer)"
    filled = [c for c in cells if c]
    has_digit = any(re.search(r"\d", c) for c in filled)
    has_flag = any(c.lower() in YES_NO for c in filled)
    short = all(len(c.split()) <= 3 for c in filled)
    if not has_digit and not has_flag and short:
        return False, "short label-only cells, no numbers or yes/no (looks like a menu)"
    return True, "ok"

# ---- OCR cleanup: drop icon glyph noise, keep only useful text ----
def clean_ocr(txt):
    txt = re.sub(r"[^\x20-\x7E]", " ", txt)           # non-ASCII = icon noise like CJK glyphs
    return re.sub(r"\s+", " ", txt).strip()

def ocr_is_useful(txt):
    words = [w for w in txt.split() if len(w) >= 3 and re.search(r"[A-Za-z]", w)]
    return len(txt) >= MIN_OCR_CHARS and len(words) >= 4

# ---- OCR (optional; tries RapidOCR new and old package, then pytesseract) ----
def make_ocr():
    import numpy as np
    from PIL import Image
    def to_np(b): return np.array(Image.open(io.BytesIO(b)).convert("RGB"))
    try:
        from rapidocr import RapidOCR
        eng = RapidOCR()
        def ocr(b):
            out = eng(to_np(b))
            txts = getattr(out, "txts", None)
            return " ".join(txts) if txts else ""
        return ocr, "rapidocr"
    except Exception:
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR
        eng = RapidOCR()
        def ocr(b):
            res, _ = eng(to_np(b))
            return " ".join(r[1] for r in res) if res else ""
        return ocr, "rapidocr_onnxruntime"
    except Exception:
        pass
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return (lambda b: pytesseract.image_to_string(Image.open(io.BytesIO(b)))), "pytesseract"
    except Exception:
        return None, "none"

if os.environ.get("PART_A_SKIP_OCR", "0") == "1":
    OCR, OCR_NAME = None, "disabled"
else:
    OCR, OCR_NAME = make_ocr()
print("OCR engine:", OCR_NAME)

def caption_image(img_bytes):
    """TODO later: short caption for diagrams/charts with a small vision model (e.g. Qwen2.5-VL).
    Returns empty string in the pilot."""
    return ""

# ======== parsers ========
import pymupdf

def parse_pdf(path):
    doc, pages, seen = pymupdf.open(path), [], set()
    for i, page in enumerate(doc, 1):
        tables, boxes, dropped = [], [], []
        try:
            for t in page.find_tables().tables:
                rows = t.extract()
                if not rows or max(len(r) for r in rows) < 2:
                    continue
                rows = clean_rows(rows)
                keep, why = table_ok(rows, t.bbox, page.rect.height)
                if keep:
                    tables.append(rows); boxes.append(pymupdf.Rect(t.bbox))
                else:
                    dropped.append(dict(page=i, reason=why, preview=table_markdown(rows)[:300]))
        except Exception:
            pass
        # text outside tables only, so table cells are not indexed twice
        blocks = []
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            centre = pymupdf.Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if any(centre in r for r in boxes):
                continue
            blocks.append(b[4].strip())
        imgs = []
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                d = doc.extract_image(xref)
                imgs.append((d["image"], d["width"], d["height"]))
            except Exception:
                continue
        pages.append(dict(page=i, text="\n\n".join(b for b in blocks if b), tables=tables, images=imgs,
                          dropped_tables=dropped))
    return pages

from pptx import Presentation

def parse_pptx(path):
    prs, pages = Presentation(path), []
    SW, SH = prs.slide_width, prs.slide_height
    for i, s in enumerate(prs.slides, 1):
        texts, tables, imgs = [], [], []
        notes = s.notes_slide.notes_text_frame.text.strip() if s.has_notes_slide else ""
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                texts.append(sh.text_frame.text.strip())
            if getattr(sh, "has_table", False) and sh.has_table:
                rows = clean_rows([[c.text for c in r.cells] for r in sh.table.rows])
                if table_ok(rows)[0]:
                    tables.append(rows)
            try:
                is_pic = sh.shape_type == 13
            except Exception:
                is_pic = False
            if is_pic:
                full_slide = sh.width * sh.height >= 0.9 * SW * SH
                if full_slide and len(notes) > 50:
                    continue            # converted slide: its text is already in the notes
                w, h = sh.image.size
                imgs.append((sh.image.blob, w, h))
        pages.append(dict(page=i, text="\n\n".join(texts + ([notes] if notes else [])),
                          tables=tables, images=imgs, dropped_tables=[]))
    return pages

# ======== ingest ========
def pick_pilot():
    if PILOT_FILES:
        return [Path(p) for p in PILOT_FILES]
    if INGEST_ALL:
        return sorted(
            p for p in CORPUS_ROOT.rglob("*")
            if p.suffix.lower() in (".pdf", ".pptx")
            and "originals_converted" not in p.parts
        )
    files = []
    for folder, n in PILOT_PER_FOLDER.items():
        d = CORPUS_ROOT / folder
        if not d.exists():
            continue
        cands = sorted(p for p in d.glob("*") if p.suffix.lower() in (".pdf", ".pptx"))
        files += random.Random(7).sample(cands, min(n, len(cands)))   # seeded: same picks every run
    for p in sorted(CORPUS_ROOT.rglob("*")):
        if p.suffix.lower() in (".pdf", ".pptx") and "originals_converted" not in p.parts \
           and any(k.lower() in p.name.lower() for k in ALSO_INCLUDE) and p not in files:
            files.append(p)
    return files

records, tables_store, dropped_store, log = [], {}, {}, []

for path in pick_pilot():
    t0, stem, ext = time.time(), path.stem, path.suffix.lower()
    url = URLS.get(stem, "")
    # Converted PPTX are page images + notes; parse the ORIGINAL PDF instead so tables/diagrams are not lost
    orig = CORPUS_ROOT / "originals_converted" / (stem + ".pdf")
    use_orig = ext == ".pptx" and orig.exists()
    try:
        pages = parse_pdf(orig) if use_orig else (parse_pdf(path) if ext == ".pdf" else parse_pptx(path))
    except Exception as e:
        log.append(dict(file=path.name, error=f"{type(e).__name__}: {e}")); continue

    alltext = " ".join(p["text"] for p in pages)
    title = pretty_title(stem)
    base = dict(doc_id=stem, title=title, tool=guess_tool(url),
                doc_type=guess_doc_type(url, ext), topic=guess_topic(alltext, title + " " + url),
                year=guess_year(url, alltext), source_url=url, sensitivity="public")
    n_text = n_tab = n_drop = img_seen = img_kept = img_ocr = 0

    for p in pages:
        for j, ch in enumerate(chunk_text(p["text"])):
            records.append(dict(id=f"{stem}::p{p['page']}::text::{j}", text=ch,
                                meta={**base, "page": p["page"], "modality": "text", "table_id": ""}))
            n_text += 1
        for d in p.get("dropped_tables", []):
            dropped_store[f"{stem}::p{d['page']}::{n_drop}"] = d; n_drop += 1
        for j, rows in enumerate(p["tables"]):
            tid = f"{stem}::p{p['page']}::table::{j}"
            tables_store[tid] = table_markdown(rows)                 # full table, fetched only if matched
            records.append(dict(id=tid, text=table_summary(rows, p["page"]) + "\n" + table_markdown(rows)[:800],
                                meta={**base, "page": p["page"], "modality": "table", "table_id": tid}))
            n_tab += 1
        for j, (blob, w, h) in enumerate(p["images"]):
            img_seen += 1
            if w < MIN_IMG_PX or h < MIN_IMG_PX or len(blob) < MIN_IMG_BYTES:
                continue
            img_kept += 1
            txt = clean_ocr(OCR(blob)) if OCR else ""
            cap = caption_image(blob)
            combined = " ".join(x for x in (cap, txt) if x)
            if ocr_is_useful(combined):
                img_ocr += 1
                records.append(dict(id=f"{stem}::p{p['page']}::image::{j}", text=combined,
                                    meta={**base, "page": p["page"], "modality": "image", "table_id": ""}))

    log.append(dict(file=path.name, format=("pptx (converted)" if use_orig else ext[1:]), tool=base["tool"],
                    topic=base["topic"], pages=len(pages), text_chunks=n_text, tables=n_tab, tables_dropped=n_drop,
                    images_seen=img_seen, images_kept=img_kept,
                    images_with_ocr_text=img_ocr, ocr_engine=OCR_NAME, seconds=round(time.time() - t0, 1), error=""))

print(f"{len(records)} chunks:", dict(Counter(r["meta"]["modality"] for r in records)))
for row in log:
    print(row)

# ======== save ========
fields = ["file", "format", "tool", "topic", "pages", "text_chunks", "tables", "tables_dropped", "images_seen",
          "images_kept", "images_with_ocr_text", "ocr_engine", "seconds", "error"]
with open(OUT / "ingest_log.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(log)
with open(OUT / "pilot_chunks.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps({"id": r["id"], "text": r["text"], **r["meta"]}, ensure_ascii=False) + "\n")
json.dump(tables_store, open(OUT / "tables_full.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(dropped_store, open(OUT / "tables_dropped.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved to", OUT)

# ======== embed ========
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=str(OUT / "chroma"))
try:
    client.delete_collection("pilot")
except Exception:
    pass
col = client.create_collection("pilot", metadata={"hnsw:space": "cosine"})

t0 = time.time()
embs = model.encode([r["text"] for r in records], normalize_embeddings=True, batch_size=32, show_progress_bar=True)
for i in range(0, len(records), 500):
    b = records[i:i + 500]
    col.add(ids=[r["id"] for r in b], documents=[r["text"] for r in b],
            embeddings=embs[i:i + 500].tolist(), metadatas=[r["meta"] for r in b])
print(f"embedded + stored {col.count()} chunks in {time.time() - t0:.1f}s")

# ======== search ========
def search(q, k=5, where=None):
    qe = model.encode(["Represent this sentence for searching relevant passages: " + q], normalize_embeddings=True)
    res = col.query(query_embeddings=qe.tolist(), n_results=k, where=where)
    for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"[{m['modality']:<5}] dist={d:.3f}  {m['doc_id'][:45]} p.{m['page']}  ({m['tool']}/{m['topic']})")
        print("   ", doc[:160].replace("\n", " ").encode("ascii", "replace").decode("ascii"))
    print()

search("How is cost calculated?")
search("What does the table show?", where={"modality": "table"})
search("architecture diagram components", where={"modality": "image"})
# metadata filter example (narrow before searching = fewer tokens later):
# search("authentication", where={"$and": [{"tool": "snowflake"}, {"modality": "text"}]})
