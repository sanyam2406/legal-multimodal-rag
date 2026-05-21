import streamlit as st

from new_rag import (
    answer_stream,
    get_collection_stats,
    ensure_indexed,
    upload_and_reindex,
    resolve_source_filter,
    resolve_prompt,
    make_message,
)
from audio_transcription import transcribe_audio, compute_audio_hash

st.set_page_config(page_title="Legal Multimodal RAG", page_icon="⚖️", layout="wide")


def init_session_state():
    if "bootstrapped" not in st.session_state:
        with st.spinner("Initializing index..."):
            ensure_indexed()
        st.session_state.bootstrapped = True

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_audio_query" not in st.session_state:
        st.session_state.pending_audio_query = None
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "_submit_audio_clicked" not in st.session_state:
        st.session_state._submit_audio_clicked = False


init_session_state()


with st.sidebar:
    st.title("Legal Multimodal RAG")
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
    selected_case = st.selectbox("Search within:", options=["All Cases"] + indexed_pdfs)
    source_filter = resolve_source_filter(selected_case)

    st.divider()
    st.subheader("Upload New PDFs")
    uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

    if uploaded and st.button("Upload & Re-index"):
        with st.spinner("Re-indexing..."):
            chunk_count, num_docs = upload_and_reindex([(f.name, f.getbuffer()) for f in uploaded])
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
            audio_hash = compute_audio_hash(audio_value.getvalue())
            if audio_hash != st.session_state.last_audio_hash:
                with st.spinner("Transcribing..."):
                    try:
                        text = transcribe_audio(audio_value)
                        st.session_state.pending_audio_query = text
                        st.session_state.last_audio_hash = audio_hash
                    except ValueError as e:
                        st.warning(str(e))
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
                    st.session_state.pending_audio_query = text
                except ValueError as e:
                    st.warning(str(e))
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
prompt, clear_audio = resolve_prompt(
    _typed_prompt,
    st.session_state._submit_audio_clicked,
    st.session_state.pending_audio_query,
)
if clear_audio:
    st.session_state.pending_audio_query = None
    st.session_state._submit_audio_clicked = False


if prompt:
    st.session_state.messages.append(make_message("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream, sources = answer_stream(prompt, source_filter)
        response = st.write_stream(stream)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append(make_message("assistant", response, sources))

