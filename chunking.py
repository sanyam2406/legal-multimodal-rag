"""
chunking.py — Ingest legal PDFs from ./data, chunk them, embed with OpenAI,
              and store in a local ChromaDB collection.

Usage:
    python chunking.py              # ingest everything in ./data
    python chunking.py --reset      # wipe the collection first, then ingest
"""

import argparse
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR         = Path("data")
CHROMA_DIR       = Path("chroma_db")
COLLECTION_NAME  = "legal_documents"
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL  = "text-embedding-3-small"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 150
SEPARATORS       = ["\n\n", "\n", ". ", " ", ""]


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    """
    Returns (full_text, first_page_text).
    full_text has [Page N] markers; first_page_text is raw for metadata extraction.
    """
    reader = PdfReader(str(pdf_path))
    pages, first = [], ""
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if raw.strip():
            pages.append(f"[Page {i}]\n{raw}")
            if i == 1:
                first = raw
    return "\n\n".join(pages), first


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    return splitter.split_text(text)


# ── Metadata extraction ────────────────────────────────────────────────────────

def extract_case_title(first_page: str) -> str:
    """
    Extract case title from the first page text.

    IndianKanoon PDFs always open with a line like:
        M/S Elora Tobacco Company Ltd. vs Union Of India on 21 July, 2022

    Strategy:
      1. Try to find 'X vs/versus Y' on the very first non-blank line.
      2. Fall back to regex scan across the whole first page.
      3. Return empty string if nothing found.
    """
    lines = [l.strip() for l in first_page.splitlines() if l.strip()]

    # Strategy 1: first line often IS the title on IndianKanoon
    for line in lines[:5]:
        m = re.search(r"(.+?)\s+(?:vs?\.?|[Vv]ersus)\s+(.+?)(?:\s+on\s+\d|\s*$)", line)
        if m:
            appellant  = m.group(1).strip().rstrip(",.").strip()
            respondent = m.group(2).strip().rstrip(",.").strip()
            if len(appellant) >= 4 and len(respondent) >= 4:
                return f"{appellant} vs {respondent}"

    # Strategy 2: broader scan
    full = re.sub(r"\s+", " ", first_page)
    m = re.search(
        r"([A-Za-z/&][A-Za-z0-9 &/\.\-\']{4,100}?)\s+(?:[Vv]s\.?|[Vv]ersus)\s+"
        r"([A-Za-z][A-Za-z0-9 &/\.\-\']{4,100})",
        full,
    )
    if m:
        appellant  = m.group(1).strip().rstrip(",.")
        respondent = m.group(2).strip().rstrip(",.")
        return f"{appellant} vs {respondent}"

    return ""


def extract_all_petition_numbers(text: str) -> list[str]:
    """
    Extract ALL petition/case numbers from a chunk of text.
    Returns a deduplicated list — needed because a single PDF can contain
    multiple writ petitions (e.g. 23618 and 23624 in the same document).
    """
    patterns = [
        r"(?:WRIT PETITION|W\.P\.?)\s*(?:\(.*?\))?\s*No\.?\s*([\w/\-]+)",
        r"(?:Civil|Criminal)\s+Appeal\s+No\.?\s*([\w/\-]+)",
        r"SLP\s*(?:\(.*?\))?\s*No\.?\s*([\w/\-]+)",
        r"Petition\s+No\.?\s*([\w/\-]+)",
        r"No\.\s*(\d{4,6}(?:/\d{4})?)",   # bare "No. 23618" or "No. 23618/2021"
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            num = m.group(1).strip()
            if num not in found:
                found.append(num)
    return found


def build_metadata(
    pdf_path: Path,
    chunk_index: int,
    total_chunks: int,
    chunk_text_str: str,
    case_title: str,
) -> dict:
    position    = round(chunk_index / max(total_chunks - 1, 1), 4)
    pet_numbers = extract_all_petition_numbers(chunk_text_str)
    # Store first detected number (most prominent) and all as pipe-separated string
    case_number  = pet_numbers[0] if pet_numbers else ""
    all_cases    = "|".join(pet_numbers)

    return {
        "source":        pdf_path.name,
        "file_path":     str(pdf_path),
        "chunk_index":   chunk_index,
        "total_chunks":  total_chunks,
        "position":      position,
        "case_number":   case_number,
        "all_cases":     all_cases,   # e.g. "23618|23624|23624/2021"
        "case_title":    case_title,
    }


# ── ChromaDB ───────────────────────────────────────────────────────────────────

def get_collection(reset: bool = False) -> chromadb.Collection:
    if not OPENAI_API_KEY:
        sys.exit("ERROR: OPENAI_API_KEY is not set.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  [reset] Deleted '{COLLECTION_NAME}'.")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: Path, collection: chromadb.Collection) -> int:
    print(f"\nProcessing: {pdf_path.name}")

    raw_text, first_page = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print("  No extractable text — skipping.")
        return 0

    case_title = extract_case_title(first_page)
    print(f"  Case title : {case_title or '(not detected)'}")

    chunks = chunk_text(raw_text)
    total  = len(chunks)
    print(f"  Chunks     : {total}")

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{pdf_path.stem}_chunk_{i:05d}")
        documents.append(chunk)
        metadatas.append(
            build_metadata(
                pdf_path,
                chunk_index=i,
                total_chunks=total,
                chunk_text_str=chunk,
                case_title=case_title,
            )
        )

    BATCH = 500
    for start in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[start:start+BATCH],
            documents=documents[start:start+BATCH],
            metadatas=metadatas[start:start+BATCH],
        )

    print(f"  Upserted {total} chunks.")
    return total


def run_ingestion(reset: bool = False) -> None:
    if not DATA_DIR.exists():
        sys.exit(f"ERROR: '{DATA_DIR}' does not exist.")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"ERROR: No PDFs in '{DATA_DIR}'.")

    print(f"Found {len(pdf_files)} PDF(s).")
    collection  = get_collection(reset=reset)
    total_chunks = 0

    for pdf_path in tqdm(pdf_files, desc="Ingesting", unit="file"):
        total_chunks += ingest_pdf(pdf_path, collection)

    print(f"\nDone — {total_chunks} chunks, {collection.count()} total in collection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run_ingestion(reset=args.reset)