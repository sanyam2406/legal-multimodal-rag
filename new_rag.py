import os
import glob
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

load_dotenv()

PDF_DIR = "data/"
DB_DIR = "chroma_db_new"
COLLECTION = "simple_rag"

 
def load_pdfs(folder):
    docs = []
    for path in glob.glob(f"{folder}/*.pdf"):
        reader = PdfReader(path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if text.strip():
            docs.append({"text": text, "source": os.path.basename(path)})
    return docs


def build_index(docs, collection):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        ids = [f"{doc['source']}_{i}" for i in range(len(chunks))]
        metas = [{"source": doc["source"]} for _ in chunks]
        collection.add(documents=chunks, metadatas=metas, ids=ids)


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
    # Remove generic question/filler words (also covers general queries like "tell about X case")
    for filler in [
        "what was", "what is", "what are", "tell me", "give me",
        "explain the", "explain", "tell about", "tell me about",
        "summarize", "summary of", "what happened in",
        " the ", " in ", " of ", " about ", " for ",
        "case", "?", ",", ".",
    ]:
        q = q.replace(filler, " ")
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
    results = collection.query(query_texts=[case_name], n_results=10)
    if not results["metadatas"][0]:
        return None
    source_counts = Counter(m["source"] for m in results["metadatas"][0])
    top_source, top_count = source_counts.most_common(1)[0]
    total = sum(source_counts.values())
    if top_count / total >= 0.4:
        return top_source
    return None


def answer(query, collection, client):
    judgment_mode = is_judgment_ratio_query(query)
    base_query = clean_retrieval_query(query)
    retrieval_query = f"{base_query} {LEGAL_ENRICHMENT}" if judgment_mode else base_query

    results = collection.query(query_texts=[retrieval_query], n_results=8)
    context = "\n\n".join(results["documents"][0])

    system_prompt = (
        "You are a legal assistant. Using only the context below, identify the "
        "binding legal principle (ratio decidendi) — the core reason why the court "
        "decided in favour of or against the parties. Focus on the final order and "
        "the court's reasoning. Do not fabricate.\n\n"
        if judgment_mode else
        "Answer using only the context below. If unsure, say so.\n\n"
    )
    
    # frontend using streamlit
    # Remove JUDGMENT_RATIO_TRIGGERS
    # Enrich base query: system_prompt
    # then test the query 

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt + context},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


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


def reindex(client_db):
    client_db.delete_collection(COLLECTION)
    ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    col = client_db.get_or_create_collection(COLLECTION, embedding_function=ef)
    docs = load_pdfs(PDF_DIR)
    build_index(docs, col)
    return col, docs


def answer_stream(query, collection, client, source_filter=None):
    judgment_mode = is_judgment_ratio_query(query)
    base_query = clean_retrieval_query(query)

    if not source_filter:
        case_name = extract_case_name_from_query(query)
        auto_source = find_document_source(case_name, collection)
        if auto_source:
            source_filter = auto_source

    if judgment_mode and source_filter:
        retrieval_query = LEGAL_ENRICHMENT
    elif judgment_mode:
        retrieval_query = f"{base_query} {LEGAL_ENRICHMENT}"
    else:
        retrieval_query = base_query

    n_results = 15 if (judgment_mode and source_filter) else 10
    query_kwargs = {"query_texts": [retrieval_query], "n_results": n_results}
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)
    context = "\n\n".join(results["documents"][0])
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"][0]))

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

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt + context},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        stream=True,
    )
    return stream, sources


def main():
    ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )
    client_db = chromadb.PersistentClient(path=DB_DIR)
    col = client_db.get_or_create_collection(COLLECTION, embedding_function=ef)

    if col.count() == 0:
        print("Indexing PDFs...")
        docs = load_pdfs(PDF_DIR)
        print(f"  Found {len(docs)} PDFs")
        build_index(docs, col)
        print(f"  Indexed {col.count()} chunks")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Ready! Type your question (or 'quit' to exit)\n")
    while True:
        q = input("Q: ").strip().replace('\n','')
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("\n\nA:", answer(q, col, openai_client), "\n")


if __name__ == "__main__":
    main()

