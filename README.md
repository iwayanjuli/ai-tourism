# AI-tourism

Basic RAG untuk edukasi wisata budaya Desa Penglipuran. Aplikasi melakukan chunking dan embedding dokumen sekali saat `build_index.py` dijalankan, menyimpan vektor di FAISS, lalu saat pertanyaan masuk hanya meng-embed pertanyaan, mengambil chunk relevan, dan meminta jawaban ke Groq `openai/gpt-oss-120b`.

## Jalankan di komputer

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env`, isi `GROQ_API_KEY` dari [Groq Console](https://console.groq.com/keys), lalu bangun indeks satu kali dengan `python build_index.py`. Setelah selesai, jalankan `streamlit run app.py`. Di PowerShell, penyalinan contoh env dapat dilakukan dengan `Copy-Item .env.example .env`; di macOS/Linux gunakan `cp .env.example .env`.

Buka alamat lokal yang ditampilkan Streamlit (biasanya http://localhost:8501).

## Deploy gratis dengan Streamlit Community Cloud

1. Buat repository GitHub dan unggah seluruh file project ini.
2. Masuk ke Streamlit Community Cloud dan hubungkan akun GitHub.
3. Pilih **Create app**, pilih repository/branch, lalu set **Main file path** ke `app.py`.
4. Deploy. Tidak ada secrets yang perlu disetel untuk versi ini.

Ketersediaan paket gratis dan langkah dashboard dapat berubah. Lihat dokumentasi resmi Streamlit untuk proses deploy: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Cara kerja RAG

Simpan dokumen sumber (`.txt`, `.md`, `.json`, atau `.pdf`) di `data/raw/`. `build_index.py` memecah teks menjadi potongan berukuran sekitar 750 karakter dengan overlap, menjalankan model embedding multilingual E5, menormalisasi vektor, dan menulis hasil chunk ke `data/processed/chunks.json` serta indeks ke `data/processed/penglipuran.faiss`. Bangun ulang indeks hanya saat konten atau model embedding berubah. Jangan menjalankan proses embedding dokumen pada setiap pertanyaan.

Saat aplikasi berjalan, FAISS mengambil empat potongan teratas. Potongan dikirim sebagai konteks ke Groq untuk menghasilkan jawaban berbahasa Indonesia yang menyertakan rujukan. Embedding dijalankan lokal memakai FastEmbed; Groq dipakai untuk generasi jawaban. Untuk deploy Streamlit Cloud, sertakan file dalam `data/processed/` yang sudah dihasilkan di repository, lalu set `GROQ_API_KEY` di bagian Secrets aplikasi. Jangan unggah `.env` ke GitHub.

Untuk JSON, gunakan satu objek atau daftar objek dengan `text` dan metadata opsional `id`, `title`, `section`, `url`. Untuk memperbarui konten, masukkan dokumen ke `data/raw/` lalu jalankan ulang `python build_index.py` secara manual. PDF yang dipindai sebagai gambar memerlukan OCR terlebih dahulu.

## Sumber materi awal

- [Profil Desa Wisata Penglipuran — Indonesia Travel](https://www.indonesia.travel/id/id/tourism-village/desa-wisata-penglipuran)
- [Tips Berkunjung dan tata ruang Penglipuran — Indonesia Travel](https://www.indonesia.travel/id/id/travel-ideas/adventure/desa-penglipuran)

Informasi operasional seperti harga tiket, jam buka, dan layanan dapat berubah; periksa kanal resmi pengelola sebelum berkunjung.
