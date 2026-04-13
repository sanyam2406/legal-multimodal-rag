import streamlit as st

from new_rag import answer_stream, get_collection_stats, save_uploaded_pdf, reindex
from audio_transcription import transcribe_audio

st.set_page_config(page_title="RAG Kanoon", page_icon="⚖️", layout="wide")


def init_resources():
    if "bootstrapped" not in st.session_state:
        chunk_count, _ = get_collection_stats()
        if chunk_count == 0:
            with st.spinner("Indexing PDFs..."):
                reindex()
        st.session_state.bootstrapped = True

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_audio_query" not in st.session_state:
        st.session_state.pending_audio_query = None
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "_submit_audio_clicked" not in st.session_state:
        st.session_state._submit_audio_clicked = False


init_resources()

with st.sidebar:
    st.title("RAG Kanoon")
    st.caption("AI Legal Research Assistant")
    st.divider()

    chunk_count, indexed_pdfs = get_collection_stats()

    st.metric("Indexed Chunks", chunk_count)
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
            save_uploaded_pdf(f.name, f.getbuffer())
        with st.spinner("Re-indexing..."):
            chunk_count, num_docs = reindex()
        st.success(f"Indexed {chunk_count} chunks from {num_docs} PDFs")
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
        stream, sources = answer_stream(prompt, source_filter)
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

