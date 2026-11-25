import csv
import json
from collections import Counter
from pathlib import Path

import pymupdf
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)


def inventory():
    rows = []
    for path in sorted(CORPUS.rglob("*")):
        ext = path.suffix.lower()
        if ext not in (".pdf", ".pptx") or "originals_converted" in path.parts:
            continue

        pages = tables = images = 0
        error = ""
        try:
            if ext == ".pdf":
                doc = pymupdf.open(path)
                pages = len(doc)
                for page in doc:
                    try:
                        tables += len(page.find_tables().tables)
                    except Exception:
                        pass
                    images += len(page.get_images(full=True))
                doc.close()
            else:
                deck = Presentation(path)
                pages = len(deck.slides)
                for slide in deck.slides:
                    for shape in slide.shapes:
                        if getattr(shape, "has_table", False):
                            tables += 1
                        if getattr(shape, "shape_type", None) == 13:
                            images += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        rows.append({
            "file": str(path.relative_to(CORPUS)).replace("\\", "/"),
            "format": ext[1:],
            "pages": pages,
            "tables": tables,
            "images": images,
            "error": error,
        })
    return rows


if __name__ == "__main__":
    rows = inventory()
    formats = Counter(r["format"] for r in rows)
    totals = {
        "distinct_documents": len(rows),
        "documents_by_format": dict(formats),
        "pages_or_slides": sum(r["pages"] for r in rows),
        "tables": sum(r["tables"] for r in rows),
        "images": sum(r["images"] for r in rows),
        "parse_errors": sum(bool(r["error"]) for r in rows),
    }
    with open(OUT / "corpus_inventory.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "format", "pages", "tables", "images", "error"])
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "corpus_inventory_summary.json", "w", encoding="utf-8") as handle:
        json.dump(totals, handle, indent=2)
    print(json.dumps(totals, indent=2))
