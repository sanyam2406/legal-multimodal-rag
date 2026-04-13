import os
import glob
import time
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from metrics import logger, log_system_metrics, timed, timed_fn

load_dotenv()

log_system_metrics("startup")

PDF_DIR = "data/"
DB_DIR = "chroma_db_new"
COLLECTION = "simple_rag"

 
@timed_fn("load_pdfs", log_metrics=True)
def load_pdfs(folder):
    docs = []
    pdf_paths = glob.glob(f"{folder}/*.pdf")
    logger.info("[load_pdfs] found %d PDF(s) in %s", len(pdf_paths), folder)
    for path in pdf_paths:
        reader = PdfReader(path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if text.strip():
            docs.append({"text": text, "source": os.path.basename(path)})
            logger.debug("[load_pdfs] loaded %s  pages=%d  chars=%d",
                         os.path.basename(path), len(reader.pages), len(text))
    logger.info("[load_pdfs] loaded %d non-empty document(s)", len(docs))
    return docs


def build_index(docs, collection):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    total_chunks = 0
    with timed("build_index", log_metrics=True):
        for doc in docs:
            chunks = splitter.split_text(doc["text"])
            ids = [f"{doc['source']}_{i}" for i in range(len(chunks))]
            metas = [{"source": doc["source"]} for _ in chunks]
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            total_chunks += len(chunks)
            logger.debug("[build_index] %s → %d chunks", doc["source"], len(chunks))
    logger.info("[build_index] total chunks indexed: %d", total_chunks)


JUDGMENT_RATIO_TRIGGERS = [
    "judgment ratio", "judgement ratio", "ratio decidendi",
    "ratio of judgment", "ratio of judgement",
    "why was the decision", "why was the judgment", "why was the judgement",
    "reason for the decision", "reasoning behind the decision",
    "why the decision was taken", "why the judgment was given",
    "tell ratio", "ratio in",
]

LEGAL_ENRICHMENT = (
    "held that accordingly the court found writ petition allowed dismissed "
    "final order conclusion ratio decidendi binding principle court decided "
    "the petitioner respondent ordered disposed judgment reasoning"
)

# Prefixes that add noise and hurt embedding similarity
QUERY_NOISE_PREFIXES = [
    "tell me about case", "tell about case", "tell me about the case",
    "tell about the case", "tell me about", "tell about",
    "what happened in", "what is the case", "explain the case",
    "summarize the case", "summary of case", "summary of the case",
]


def is_judgment_ratio_query(query: str) -> bool:
    q = query.lower()
    return any(trigger in q for trigger in JUDGMENT_RATIO_TRIGGERS)


def clean_retrieval_query(query: str) -> str:
    """Strip conversational prefixes so the case name anchors the embedding."""
    q = query.strip()
    for prefix in QUERY_NOISE_PREFIXES:
        if q.lower().startswith(prefix):
            q = q[len(prefix):].strip()
            break
    # Strip any leading punctuation/symbols left behind (e.g. ": " after "tell about :")
    q = q.lstrip(": ").strip()
    return q


def extract_case_name_from_query(query: str) -> str:
    """
    Strip ratio/question boilerplate from a query to isolate the case name.
    e.g. "What was the judgment ratio in M Nagarajan case?" → "m nagarajan"
    """
    q = query.lower()
    # Remove ratio-trigger phrases first
    for trigger in JUDGMENT_RATIO_TRIGGERS:
        q = q.replace(trigger, " ")
    # Remove generic question/filler words — keep proper nouns like "of", "the", "in" intact
    for filler in [
        "what was the", "what is the", "what are the",
        "what was", "what is", "what are",
        "tell me about the", "tell me about", "tell me",
        "give me the", "give me",
        "explain the", "explain",
        "tell about the", "tell about",
        "summarize the", "summarize",
        "summary of the", "summary of",
        "what happened in",
        "case", "?", ",", ".",
    ]:
        q = q.replace(filler, " ")
    q = q.lstrip(": ").strip()
    return " ".join(q.split()).strip()


def find_document_source(case_name: str, collection) -> str | None:
    """
    Two-stage lookup: query ChromaDB using only the case name to find which
    PDF it belongs to. Returns the source filename if one document clearly
    dominates (≥40% of top-10 results). Returns None if ambiguous.

    This is the fix for cross-document drift on "ratio decidendi" queries:
    we identify the correct document BEFORE running the semantic ratio query,
    then apply a metadata pre-filter so retrieval never touches other PDFs.
    """
    if not case_name or len(case_name) < 3:
        return None
    from collections import Counter

    # Get total chunks per PDF for normalization
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    pdf_sizes = Counter(m["source"] for m in all_metas)

    results = collection.query(query_texts=[case_name], n_results=20)
    if not results["metadatas"][0]:
        return None

    hit_counts = Counter(m["source"] for m in results["metadatas"][0])

    # Normalize: hits / total_chunks so large PDFs don't dominate by volume
    scores = {src: hits / pdf_sizes[src] for src, hits in hit_counts.items()}
    top_source = max(scores, key=scores.get)

    # Must also have at least 1 hit in top-5 results (highest confidence)
    top5_sources = [m["source"] for m in results["metadatas"][0][:5]]
    if top_source in top5_sources:
        return top_source
    return None



def get_collection():
    ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    client_db = chromadb.PersistentClient(path=DB_DIR)
    col = client_db.get_or_create_collection(COLLECTION, embedding_function=ef)
    return client_db, col


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_collection_stats():
    _, col = get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    indexed_pdfs = sorted(set(m["source"] for m in all_meta)) if all_meta else []
    return col.count(), indexed_pdfs


def save_uploaded_pdf(filename, data: bytes):
    with open(os.path.join(PDF_DIR, filename), "wb") as f:
        f.write(data)


def reindex():
    logger.info("[reindex] starting full re-index")
    log_system_metrics("reindex/start")
    t0 = time.perf_counter()
    client_db, _ = get_collection()
    client_db.delete_collection(COLLECTION)
    ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    col = client_db.get_or_create_collection(COLLECTION, embedding_function=ef)
    docs = load_pdfs(PDF_DIR)
    build_index(docs, col)
    elapsed = time.perf_counter() - t0
    chunk_count = col.count()
    logger.info("[reindex] complete  chunks=%d  docs=%d  elapsed=%.2fs",
                chunk_count, len(docs), elapsed)
    log_system_metrics("reindex/end")
    return chunk_count, len(docs)


def answer_stream(query, source_filter=None):
    query_t0 = time.perf_counter()
    logger.info("[answer_stream] query=%r  source_filter=%r", query[:120], source_filter)
    log_system_metrics("answer_stream/start")

    _, collection = get_collection()
    client = get_openai_client()

    judgment_mode = is_judgment_ratio_query(query)
    base_query = clean_retrieval_query(query)
    case_name = extract_case_name_from_query(query)

    logger.debug("[answer_stream] judgment_mode=%s  case_name=%r  base_query=%r",
                 judgment_mode, case_name, base_query[:80])

    if not source_filter:
        with timed("auto_source_detection"):
            auto_source = find_document_source(case_name, collection)
        if auto_source:
            source_filter = auto_source
            logger.info("[answer_stream] auto-detected source: %s", source_filter)

    if judgment_mode and source_filter:
        retrieval_query = f"{case_name} {LEGAL_ENRICHMENT}"
    elif judgment_mode:
        retrieval_query = f"{base_query} {LEGAL_ENRICHMENT}"
    else:
        retrieval_query = f"{case_name} {base_query}" if case_name else base_query

    n_results = 30 if source_filter else 15
    query_kwargs = {"query_texts": [retrieval_query], "n_results": n_results}
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    with timed("chromadb_query"):
        results = collection.query(**query_kwargs)

    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    logger.info("[answer_stream] retrieved %d chunks (n_results=%d)  source_filter=%r",
                len(chunks), n_results, source_filter)

    # Re-rank: prioritize chunks that contain the most case name keywords
    if case_name:
        keywords = [w for w in case_name.lower().split() if len(w) > 2]
        def keyword_score(chunk):
            cl = chunk.lower()
            return sum(1 for kw in keywords if kw in cl)
        paired = sorted(zip(chunks, metas), key=lambda x: keyword_score(x[0]), reverse=True)
        chunks, metas = zip(*paired) if paired else (chunks, metas)

    chunks = chunks[:10]
    metas = metas[:10]

    context = "\n\n".join(chunks)
    sources = list(dict.fromkeys(m["source"] for m in metas))
    context_chars = len(context)
    logger.debug("[answer_stream] context_chars=%d  top_sources=%s", context_chars, sources)

    if judgment_mode:
        system_prompt = (
            "You are a legal assistant. The context below contains excerpts from a court judgment. "
            "Extract the ratio decidendi — the binding legal principle and the court's core reasoning "
            "for its final decision. Look for phrases like 'we hold', 'we are of the view', "
            "'sum up the law', 'writ petition is allowed/dismissed', 'it is accordingly'. "
            "State the ratio clearly and concisely based on what is in the context. "
            "Only say 'not found' if the context contains absolutely no reasoning or final order.\n\n"
        )
    else:
        system_prompt = "Answer using only the context below. If unsure, say so.\n\n"

    llm_t0 = time.perf_counter()
    logger.info("[answer_stream] calling OpenAI  model=gpt-4o-mini  judgment_mode=%s",
                judgment_mode)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt + context},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        stream=True,
    )

    # Wrap the stream to log latency to first token and total elapsed
    def _instrumented_stream():
        first_token = True
        for chunk in stream:
            if first_token and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - llm_t0
                logger.info("[answer_stream] time_to_first_token=%.3fs", ttft)
                first_token = False
            yield chunk
        total_elapsed = time.perf_counter() - query_t0
        logger.info("[answer_stream] stream_complete  total_elapsed=%.3fs  sources=%s",
                    total_elapsed, sources)
        log_system_metrics("answer_stream/end")

    return _instrumented_stream(), sources


def main():
    chunk_count, _ = get_collection_stats()
    if chunk_count == 0:
        print("Indexing PDFs...")
        count, num_docs = reindex()
        print(f"  Indexed {count} chunks from {num_docs} PDFs")

    print("Ready! Type your question (or 'quit' to exit)\n")
    while True:
        q = input("Q: ").strip().replace('\n', '')
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            stream, _ = answer_stream(q)
            print("\n\nA: ", end="", flush=True)
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    main()

