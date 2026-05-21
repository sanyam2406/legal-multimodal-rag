# Legal Multimodal RAG ⚖️
> A multimodal legal research assistant powered by Retrieval-Augmented Generation

Legal Multimodal RAG lets you query a corpus of Indian legal PDFs (including scanned documents) using natural language — by text or voice. It retrieves the most relevant precedents, helps identify key judgment ratios, and streams answers with source attribution.

## Features

- **Semantic Case Search** — Retrieves up to 30 relevant chunks per query, re-ranked by keyword match, then answered by `gpt-4o-mini`
- **Voice Queries** — Record via microphone or upload audio (WAV/MP3/OGG/FLAC); transcribed locally using Whisper (`faster-whisper`, CPU, int8)
- **Multimodal Ingestion** — Three-tier PDF extraction: direct text → scanned page OCR (300 DPI) → embedded image OCR
- **Ratio Decidendi Detection** — Automatically identifies judgment ratio queries and applies legal keyword enrichment
- **Case-Scoped Search** — Filter answers to a specific indexed PDF to prevent cross-document hallucination
- **Live Document Ingestion** — Upload new PDFs from the sidebar; full re-index without restart
- **Observability** — Structured logging with TTFT (time-to-first-token), CPU/memory snapshots, and operation timing via decorators

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM | OpenAI `gpt-4o-mini` (temperature 0.2) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB |
| RAG Orchestration | LangChain |
| Audio Transcription | faster-whisper (Whisper `base`, CPU, int8) |
| PDF Extraction | PyMuPDF (fitz) |
| OCR | Tesseract via pytesseract + Pillow |
| Metrics | psutil + Python logging |

---

## Architecture

```
User Query (text or voice)
        │
        ▼
 Query Preprocessing
 (case name extraction, noise stripping)
        │
        ▼
 ChromaDB Semantic Search
 (text-embedding-3-small, top-30 chunks)
        │
        ▼
 Keyword Re-ranking → Top-10 context assembly
        │
        ▼
 gpt-4o-mini (streaming, temperature=0.2)
        │
        ▼
 Response + Source Attribution
```

**Document ingestion pipeline:**
```
PDF / Image
    │
    ├── Direct text extraction (PyMuPDF, ≥50 chars/page)
    ├── Scanned page OCR    (300 DPI render → Tesseract)
    └── Embedded image OCR  (Tesseract, ≥100px dimensions)
        │
        ▼
 RecursiveCharacterTextSplitter (chunk=1000, overlap=150)
        │
        ▼
 ChromaDB (text-embedding-3-small)
```

---

## Installation

### Prerequisites
- Python 3.10+
- Tesseract OCR engine

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

### Setup

```bash
git clone https://github.com/sanyam2406/legal-multimodal-rag.git
cd legal-multimodal-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional — model override
WHISPER_MODEL=base          # tiny | base | small | medium
```

---

## Usage

```bash
streamlit run app.py
```

1. Open `http://localhost:8501`
2. Drop legal PDFs in the **sidebar uploader** and click **Upload & Re-index**
3. Ask questions in the chat box or record via the **🎙️ voice panel**
4. Use the **Filter by Case** dropdown to scope answers to a specific document

---

## Project Structure

```
legal-multimodal-rag/
├── app.py                  # Streamlit entry point
├── new_rag.py              # Core RAG engine (indexing, retrieval, streaming)
├── audio_transcription.py  # Whisper transcription
├── pdf_extractor.py        # Three-tier PDF/image extraction
├── ocr_engine.py           # Tesseract OCR wrapper with graceful fallback
├── metrics.py              # Structured logging + system metrics
├── requirements.txt
├── .env.example
├── data/                   # PDF corpus (gitignored)
├── chroma_db/              # Vector store (gitignored)
└── logs/                   # Runtime logs (gitignored)
```

---

## Future Improvements

- [ ] Support for multilingual queries (Hindi, Urdu) via Whisper language detection
- [ ] Citation pinpointing — link answers to exact page numbers
- [ ] BM25 hybrid retrieval alongside dense embeddings
- [ ] FastAPI backend with REST endpoints for headless integration
- [ ] Docker image for one-command deployment
- [ ] User authentication for multi-tenant document access
