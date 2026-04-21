#!/usr/bin/env python3
"""Export Felleskjopet SQLite pages (+ sqlite-vec embeddings) to ProductDocument JSONL.

Usage:
    python db_to_jsonl.py <input.db> <output.jsonl>

Emits all rows where `embedded=1` (has a vector in vec_pages).
"""
import json
import re
import sqlite3
import struct
import sys
from urllib.parse import urlparse

import sqlite_vec


VECTOR_DIM = 3072


def parse_price(raw: str) -> float:
    if not raw:
        return 0.0
    # Norwegian prices may use comma decimals; keep digits, comma, dot, minus
    cleaned = re.sub(r"[^0-9,.\-]", "", raw).replace(",", ".")
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def category_from_url(url: str, fallback: str) -> str:
    try:
        path = urlparse(url).path.strip("/")
        parts = path.split("/")
        # .../produkt/<cat>/<subcat>/<slug> or .../artikler/<cat>/...
        if parts and parts[0] in ("produkt", "artikler") and len(parts) >= 3:
            return " / ".join(parts[1:-1])
    except Exception:
        pass
    return fallback or ""


def unpack_vector(blob: bytes) -> list:
    if len(blob) != VECTOR_DIM * 4:
        raise ValueError(f"vector blob is {len(blob)} bytes, expected {VECTOR_DIM * 4}")
    return list(struct.unpack(f"<{VECTOR_DIM}f", blob))


def main(db_path: str, out_path: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()

    total = cur.execute(
        "SELECT COUNT(*) FROM pages WHERE embedded=1"
    ).fetchone()[0]
    print(f"[export] {total} enriched rows to process", flush=True)

    rows = cur.execute(
        """
        SELECT p.id, p.url, p.page_type, p.title, p.meta_description,
               p.breadcrumbs, p.beskrivelse, p.product_code, p.brand,
               p.price, p.ai_synonyms, p.ai_decompounded, p.ai_use_cases,
               p.ai_keywords, v.embedding
          FROM pages p
          JOIN vec_pages v ON v.page_id = p.id
         WHERE p.embedded = 1
        """
    )

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for r in rows:
            (pid, url, page_type, title, meta_desc, breadcrumbs, beskrivelse,
             product_code, brand, price, ai_syn, ai_dec, ai_uc, ai_kw, emb) = r

            try:
                vector = unpack_vector(emb)
            except ValueError as e:
                print(f"[export] skip id={pid}: {e}", flush=True)
                skipped += 1
                continue

            product_id = (product_code or "").strip() or f"page-{pid}"

            doc = {
                "product_id": product_id,
                "name": (title or "").strip(),
                "brand": (brand or "").strip(),
                "category": category_from_url(url or "", page_type or ""),
                "sku": (product_code or "").strip(),
                "color": "",
                "size": "",
                "material": "",
                "description": (meta_desc or "").strip(),
                "specifications": (beskrivelse or "").strip(),
                "price": parse_price(price or ""),
                "stock": 0,
                "attributes": {
                    "url": url or "",
                    "page_type": page_type or "",
                    "ai_synonyms": ai_syn or "",
                    "ai_decompounded": ai_dec or "",
                    "ai_use_cases": ai_uc or "",
                    "ai_keywords": ai_kw or "",
                },
                "embedding": vector,
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"[export] {written}/{total}", flush=True)

    print(f"[export] done: {written} written, {skipped} skipped -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: db_to_jsonl.py <input.db> <output.jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
