"""Read source files from data/raw, chunk them into data/processed, and build FAISS."""
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
load_dotenv(ROOT / ".env")
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 120
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def _metadata(path, values=None):
    """Normalize optional metadata, with sensible defaults from the filename."""
    values = values or {}
    return {
        "source": str(path.relative_to(RAW_DIR)).replace("\\", "/"),
        "title": values.get("title") or path.stem.replace("_", " ").title(),
        "section": values.get("section") or "Materi wisata budaya",
        "url": values.get("url") or "",
        "document_id": str(values.get("id") or re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem)),
    }


def load_documents():
    """Load TXT/Markdown, JSON records, and PDF pages found under data/raw."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Folder sumber belum ditemukan: {RAW_DIR}")

    files = [p for p in sorted(RAW_DIR.rglob("*"))
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise FileNotFoundError(
            f"Tidak ada dokumen di {RAW_DIR}. Tambahkan file .txt, .md, .json, atau .pdf."
        )

    documents = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8-sig").strip()
            if text:
                documents.append({**_metadata(path), "text": text})
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = data if isinstance(data, list) else data.get("documents", [data]) if isinstance(data, dict) else []
            for number, row in enumerate(rows, start=1):
                if isinstance(row, str):
                    row = {"text": row}
                if not isinstance(row, dict) or not str(row.get("text", "")).strip():
                    continue
                metadata = _metadata(path, row)
                if len(rows) > 1 and not row.get("id"):
                    metadata["document_id"] += f"-{number}"
                documents.append({**metadata, "section": row.get("section") or metadata["section"],
                                  "title": row.get("title") or metadata["title"],
                                  "url": row.get("url") or metadata["url"], "text": row["text"].strip()})
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Untuk membaca PDF, install dependensi: pip install pypdf") from exc
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    meta = _metadata(path)
                    documents.append({**meta, "section": f"Halaman {page_number}", "text": text})

    if not documents:
        raise ValueError("Dokumen ditemukan, tetapi tidak ada teks yang dapat diproses.")
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text by words into chunks bounded by characters with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start
        current = []
        while end < len(words):
            candidate = " ".join(current + [words[end]])
            if len(candidate) > chunk_size and current:
                break
            current.append(words[end])
            end += 1
        chunks.append(" ".join(current))
        if end >= len(words):
            break
        overlap_count = max(1, len(current) * overlap // max(len(" ".join(current)), 1))
        start = max(start + 1, end - overlap_count)
    return chunks


def build_chunks(documents):
    records = []
    for document in documents:
        for number, text in enumerate(chunk_text(document["text"]), start=1):
            records.append({
                "id": f"{document['document_id']}-C{number:02d}",
                "title": document["title"], "section": document["section"],
                "url": document["url"], "source": document["source"], "text": text,
            })
    return records


def main():
    global MODEL_NAME
    supported_models = {item["model"] for item in TextEmbedding.list_supported_models()}
    if MODEL_NAME not in supported_models:
        print(f"Model embedding '{MODEL_NAME}' tidak didukung versi FastEmbed ini.")
        MODEL_NAME = DEFAULT_MODEL_NAME
        print(f"Menggunakan model yang didukung: {MODEL_NAME}")
    documents = load_documents()
    records = build_chunks(documents)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Simpan hasil chunk sebelum embedding agar bisa diperiksa dan dipakai ulang.
    chunks_path = PROCESSED_DIR / "chunks.json"
    chunks_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Membuat embedding untuk {len(records)} chunk...")
    model = TextEmbedding(model_name=MODEL_NAME)
    vectors = np.asarray(list(model.embed([item["text"] for item in records])), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(PROCESSED_DIR / "penglipuran.faiss"))
    print(f"Selesai: {len(documents)} dokumen menjadi {len(records)} chunk.")
    print(f"Chunk: {chunks_path}")
    print(f"Indeks FAISS: {PROCESSED_DIR / 'penglipuran.faiss'}")


if __name__ == "__main__":
    main()
