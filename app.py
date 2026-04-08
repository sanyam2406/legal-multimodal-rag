import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from new_rag import (
    load_pdfs, build_index,
    is_judgment_ratio_query, clean_retrieval_query,
    extract_case_name_from_query, find_document_source,
    PDF_DIR, DB_DIR, COLLECTION, LEGAL_ENRICHMENT,
)
from audio_transcription import transcribe_audio
# api contract 
# why r we splitting the task of backend n frontend in production 
# why we need api's y not smth else 
# restful api 

load_dotenv()

st.set_page_config(page_title="RAG Kanoon", page_icon="⚖️", layout="wide")

def init_resources():
    if "col" not in st.session_state:
        ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
        )

        client_db = chromadb.PersistentClient(path=DB_DIR)
        col = client_db.get_or_create_collection(COLLECTION, embedding_function=ef)
        
        if col.count() == 0:
            # TODO: Add documents to collection
            with st.spinner("Indexing PDFs..."):
                docs = load_pdfs(PDF_DIR)
                build_index(docs, col)
        
        st.session_state.client_db = client_db
        st.session_state.col = col
        st.session_state.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_audio_query" not in st.session_state:
        st.session_state.pending_audio_query = None
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "_submit_audio_clicked" not in st.session_state:
        st.session_state._submit_audio_clicked = False

init_resources()
        
# to add : name and date as a metadata to chunk 
# prompt 

def answer_stream(query, collection, client, source_filter=None):
    judgment_mode = is_judgment_ratio_query(query)
    base_query = clean_retrieval_query(query)
    retrieval_query = f"{base_query} {LEGAL_ENRICHMENT}" if judgment_mode else base_query

    # --- Pre-filter: identify the correct document BEFORE semantic search ---
    # For ratio/decidendi queries the embedding drifts toward generic legal
    # language and retrieves chunks from the wrong PDFs. We fix this by doing
    # a separate case-name-only lookup first, then restricting retrieval to
    # that document via a metadata filter.
    if not source_filter and judgment_mode:
        case_name = extract_case_name_from_query(query)
        auto_source = find_document_source(case_name, collection)
        if auto_source:
            source_filter = auto_source

    query_kwargs = {"query_texts": [retrieval_query], "n_results": 8}
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)
    context = "\n\n".join(results["documents"][0])
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"][0]))

    if judgment_mode:
        system_prompt = (
            "You are a legal assistant. Using ONLY the context below, identify the "
            "binding legal principle (ratio decidendi) — the core reason why the court "
            "decided in favour of or against the parties. Focus on the final order and "
            "the court's reasoning. Do not fabricate. "
            "If the context does not contain enough information to answer, say: "
            "'The ratio decidendi for this case was not found in the indexed documents.'\n\n"
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

with st.sidebar:
    st.title("RAG Kanoon")
    st.caption("AI Legal Research Assistant")
    st.divider()

    col = st.session_state.col
    all_meta = col.get(include=["metadatas"])["metadatas"]
    indexed_pdfs = sorted(set(m["source"] for m in all_meta)) if all_meta else []

    st.metric("Indexed Chunks", col.count())
    st.metric("Documents", len(indexed_pdfs))

    with st.expander("Indexed PDFs"):
        for pdf in indexed_pdfs:
            st.text(pdf)

    st.divider()
    st.subheader("Filter by Case")
    selected_case = st.selectbox(
        "Search within:",
        options=["All Cases"] + indexed_pdfs,
    )
    source_filter = None if selected_case == "All Cases" else selected_case

    st.divider()
    st.subheader("Upload New PDFs")
    uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

    if uploaded and st.button("Upload & Re-index"):
        for f in uploaded:
            with open(os.path.join(PDF_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())

        st.session_state.client_db.delete_collection(COLLECTION)
        ef = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        new_col = st.session_state.client_db.get_or_create_collection(COLLECTION, embedding_function=ef)
        with st.spinner("Re-indexing..."):
            docs = load_pdfs(PDF_DIR)
            build_index(docs, new_col)
        st.session_state.col = new_col
        st.success(f"Indexed {new_col.count()} chunks from {len(docs)} PDFs")
        st.rerun()

st.header("Legal Research Chat")

def _on_submit_audio():
    st.session_state._submit_audio_clicked = True

with st.expander("🎙️ Ask by voice", expanded=False):
    tab_mic, tab_file = st.tabs(["Microphone", "Upload Audio File"])

    with tab_mic:
        audio_value = st.audio_input("Record your legal question", key="mic_input")
        if audio_value is not None:
            audio_hash = hash(audio_value.getvalue())
            if audio_hash != st.session_state.last_audio_hash:
                with st.spinner("Transcribing..."):
                    try:
                        text = transcribe_audio(audio_value)
                        if text:
                            st.session_state.pending_audio_query = text
                            st.session_state.last_audio_hash = audio_hash
                        else:
                            st.warning("No speech detected. Please try again.")
                    except RuntimeError as e:
                        st.error(str(e))

    with tab_file:
        uploaded_audio = st.file_uploader(
            "Upload an audio file", type=["wav", "mp3", "ogg", "flac"],
            key="audio_file_input",
        )
        if uploaded_audio and st.button("Transcribe File", key="transcribe_file_btn"):
            with st.spinner("Transcribing..."):
                try:
                    text = transcribe_audio(uploaded_audio)
                    if text:
                        st.session_state.pending_audio_query = text
                    else:
                        st.warning("No speech detected.")
                except RuntimeError as e:
                    st.error(str(e))

    if st.session_state.pending_audio_query:
        st.info(f"Transcribed: **{st.session_state.pending_audio_query}**")
        c1, c2 = st.columns(2)
        with c1:
            st.button("Submit this question", type="primary",
                      key="submit_audio_btn", on_click=_on_submit_audio)
        with c2:
            if st.button("Clear", key="clear_audio_btn"):
                st.session_state.pending_audio_query = None
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

_typed_prompt = st.chat_input("Ask a legal question...")
prompt = _typed_prompt

if (not prompt
        and st.session_state.get("_submit_audio_clicked")
        and st.session_state.get("pending_audio_query")):
    prompt = st.session_state.pending_audio_query
    st.session_state.pending_audio_query = None
    st.session_state._submit_audio_clicked = False

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream, sources = answer_stream(
            prompt, st.session_state.col, st.session_state.openai_client, source_filter
        )
        response = st.write_stream(
            chunk.choices[0].delta.content
            for chunk in stream
            if chunk.choices[0].delta.content is not None
        )
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

        
        
        
        
