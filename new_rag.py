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
    "final decision petition allowed dismissed final order of court "
    "writ allowed disposed held that accordingly ordered judgment"
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

