import os
import dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

dotenv.load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

DATA_PATH = "./data"
DB_PATH = "./db"


# 1) Load PDFs
loader = PyPDFDirectoryLoader(DATA_PATH)
docs = loader.load()

pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
print("Number of PDFs:", len(pdf_files))
print("PDF files:", pdf_files)
print("Number of documents loaded:", len(docs))

if docs:
    print("First document sample:", docs[0].page_content[:500])

# Extract case names from all documents (generic)
case_names = set()
case_to_source = {}
import re

for doc in docs:
    source = doc.metadata.get("source", "")
    # Look for "vs" patterns in documents
    matches = re.findall(r'([A-Za-z\s]+)\s+vs\s+([A-Za-z\s]+)', doc.page_content[:2000])
    for match in matches:
        case_name = f"{match[0].strip()} vs {match[1].strip()}"
        case_names.add(case_name)
        case_to_source[case_name] = source

print(f"Found {len(case_names)} case names:")
for case in list(case_names)[:5]:  # Show first 5
    print(f"  - {case}")


# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)
chunks = splitter.split_documents(docs)
print("Number of chunks created:", len(chunks))


# 3) Embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)


# 4) Create or load Chroma DB
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print("Loading existing Chroma DB...")
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
else:
    print("Creating new Chroma DB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )


# 5) LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_KEY
)


# 6) Detect query type
def detect_query_type(question):
    q = question.lower()

    if any(word in q for word in [
        "final judgment", "final order", "judgment", "order",
        "allowed", "dismissed", "disposed", "decision", "held"
    ]):
        return "judgment"

    if any(word in q for word in [
        "why", "reason", "reasoning", "ratio", "ratio decidendi",
        "because", "rationale", "legal reasoning", "principle"
    ]):
        return "reasoning"

    if any(word in q for word in [
        "court", "date", "judge", "bench", "coram"
    ]):
        return "metadata"

    if any(word in q for word in [
        "petitioner", "respondent", "appellant", "plaintiff", "defendant", "parties"
    ]):
        return "parties"

    return "general"


# 7) Main RAG function
def find_case_source(question):
    q = question.lower()
    
    # 1. Try PDF number first (most reliable)
    pdf_numbers = re.findall(r'(\d{8})\.pdf', q)
    if pdf_numbers:
        return f"data/{pdf_numbers[0]}.pdf"
    
    # 2. Try generic case name matching
    print(f"DEBUG: Available case names: {list(case_names)[:3]}...")  # Show first 3
    for case_name in case_names:
        if case_name.lower() in q:
            source = case_to_source.get(case_name)
            print(f"DEBUG: Found case name '{case_name}' -> {source}")
            return source
    
    # 3. Try partial case name matching
    for case_name in case_names:
        case_parts = case_name.lower().split(" vs ")
        if any(part in q for part in case_parts):
            source = case_to_source.get(case_name)
            print(f"DEBUG: Found partial case name '{case_name}' -> {source}")
            return source
    
    print("DEBUG: No case source found")
    return None


def ask_question(question):
    query_type = detect_query_type(question)

    # First try to identify exact case PDF (generic for all PDFs)
    forced_source = find_case_source(question)

    if query_type == "reasoning":
        retrieval_query = (
            f"ratio decidendi reasoning rationale legal basis findings "
            f"why court held principle applied jurisdiction mortgage bank priority "
            f"{question}"
        )
    elif query_type == "judgment":
        retrieval_query = (
            f"final judgment order decision allowed dismissed disposed held "
            f"outcome result set aside {question}"
        )
    elif query_type == "metadata":
        retrieval_query = f"court date judge bench coram {question}"
    elif query_type == "parties":
        retrieval_query = f"petitioner respondent appellant plaintiff defendant parties {question}"
    else:
        retrieval_query = question

    # Get more chunks for better context
    results = vectordb.similarity_search_with_score(retrieval_query, k=15)

    if not results:
        return "No relevant chunks found.", []

    # If we found a likely case PDF, force retrieval to that source
    if forced_source:
        best_source = forced_source
    else:
        best_source = results[0][0].metadata.get("source")

    same_source = [
        (doc, score) for doc, score in results
        if doc.metadata.get("source") == best_source
    ]

    if query_type == "judgment":
        filtered_results = sorted(
            same_source,
            key=lambda x: x[0].metadata.get("page", 0),
            reverse=True
        )[:4]

    elif query_type == "metadata":
        filtered_results = sorted(
            same_source,
            key=lambda x: x[0].metadata.get("page", 0)
        )[:3]

    elif query_type == "parties":
        filtered_results = sorted(
            same_source,
            key=lambda x: x[0].metadata.get("page", 0)
        )[:3]

    elif query_type == "reasoning":
        filtered_results = sorted(
            same_source,
            key=lambda x: x[0].metadata.get("page", 0)
        )[:7]

    else:
        filtered_results = same_source[:4]

    print(f"\n--- Filtered Chunks ({query_type.upper()} | Same Case Only) ---")
    print("Main case PDF:", best_source)

    retrieved_docs = []
    for i, (doc, score) in enumerate(filtered_results):
        retrieved_docs.append(doc)
        print(f"\nChunk {i+1}")
        print("Score:", score)
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print(doc.page_content[:1200])
        print("-" * 80)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    if query_type == "reasoning":
     prompt = f"""
You are analyzing a legal judgment.

Task:
Explain the reasoning behind the judgment using only the provided context.

Rules:
1. Use only the provided context to answer.
2. Extract relevant information from the text.
3. Be helpful and direct.
4. If specific information is not found, say what you can find.

Context:
{context}

Question: {question}

Answer:
"""
    else:
        prompt = f"""
You are answering from the provided legal document context.

Rules:
1. Use only the provided context to answer.
2. Extract relevant information from the text.
3. Be helpful and direct.
4. If specific information is not found, say what you can find.

Context:
{context}

Question: {question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content, retrieved_docs


# 8) CLI loop
while True:
    q = input("\nAsk: ")
    if q.lower().strip() in ["exit", "quit"]:
        break

    answer, retrieved_docs = ask_question(q)
    print("\nAnswer:", answer)

    print("\nSources used:")
    seen = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        print(f"- {source} | page {page}")
        seen.add(source)

    print("\nCase found in PDF file(s):")
    for source in seen:
        print(f"- {source}")