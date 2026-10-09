"""RAG runtime: load prebuilt FAISS index, embed only the query, generate with Groq."""
import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
INDEX_PATH = ROOT / "data" / "processed" / "penglipuran.faiss"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@st.cache_resource(show_spinner="Memuat model embedding dan indeks...")
def load_retriever():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("Indeks belum dibuat. Jalankan `python build_index.py` terlebih dahulu.")
    supported_models = {item["model"] for item in TextEmbedding.list_supported_models()}
    model_name = EMBEDDING_MODEL
    if model_name not in supported_models:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        st.warning(f"Model embedding di .env tidak didukung FastEmbed; memakai {model_name}.")
    model = TextEmbedding(model_name=model_name)
    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return model, index, chunks


def retrieve(question, top_k=4):
    model, index, chunks = load_retriever()
    query_vector = np.asarray(list(model.embed([question]))[0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    return [(float(score), chunks[int(i)]) for score, i in zip(scores[0], indices[0]) if i >= 0]


def answer_question(question):
    hits = retrieve(question)
    if not hits or hits[0][0] < 0.25:
        return {"answer": "Saya belum menemukan informasi yang cukup pada materi Penglipuran yang tersedia. Coba tanyakan tentang lokasi, tata ruang, angkul-angkul, atau hutan bambu.", "sources": []}

    context = "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for _, chunk in hits)
    citations = "\n".join(f"[{chunk['id']}] {chunk['title']} — {chunk['section']} ({chunk['url']})" for _, chunk in hits)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            raise RuntimeError("GROQ_API_KEY belum diatur. Isi file .env lokal atau Secrets di Streamlit Cloud.")

    response = Groq(api_key=api_key).chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Anda adalah pemandu edukasi budaya Penglipuran. Jawab dalam bahasa Indonesia dengan ramah. Gunakan hanya konteks yang diberikan; bila konteks tidak cukup, katakan demikian. Jangan mengarang fakta. Sertakan penanda sumber seperti [P1] pada klaim yang relevan. Hormati masyarakat dan adat setempat."},
            {"role": "user", "content": f"Pertanyaan: {question}\n\nKonteks:\n{context}\n\nReferensi:\n{citations}"},
        ],
        temperature=0.2,
        max_completion_tokens=800,
    )
    sources = [{"id": chunk["id"], "title": chunk["title"], "section": chunk["section"], "excerpt": chunk["text"], "url": chunk["url"], "score": round(score, 3)} for score, chunk in hits]
    return {"answer": response.choices[0].message.content or "Maaf, saya belum dapat menyusun jawaban.", "sources": sources}
