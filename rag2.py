from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

import os

load_dotenv()

# Load OpenAI key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load PDFs from data/
loader = PyPDFDirectoryLoader("./data")
docs = loader.load()

pdf_files = [f for f in os.listdir("./data") if f.endswith(".pdf")]
print("Number of PDFs:", len(pdf_files))
print("PDF files:", pdf_files)

print("Number of documents loaded:", len(docs))
if docs:
    print("First document sample:", docs[0].page_content[:500])



# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
print("Number of chunks created:", len(chunks))

# 3. Create embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)

# 4. Vector DB
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

# 5. Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# 6. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 7. Simple RAG function
def ask_question(question):
    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(question)
    
    # Prepare context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt
    prompt = f"""Answer the following question based only on the provided context:

Context:
{context}

Question: {question}

Answer:"""
    
    # Get answer from LLM
    response = llm.invoke(prompt)
    
    return response.content, relevant_docs

# i need to check what are the chunk that get pulled for each question   

# 8. Ask questions
while True:
    q = input("\nAsk: ")
    if q.lower().strip() in ["exit", "quit"]:
        break
    answer, chunks = ask_question(q)
    print("\nAnswer:", answer)
    
    # Show retrieved chunks
    print("\n--- Retrieved Chunks ---")
    for i, doc in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:300]}...")
        print("-" * 50)


