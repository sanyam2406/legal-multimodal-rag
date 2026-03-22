"""
main.py — Interactive chat loop for the Legal RAG pipeline.

Run:
    python main.py

Prerequisites:
  1. OPENAI_API_KEY set in environment
  2. python chunking.py --reset   (re-ingest after chunking.py changes)

Commands inside chat:
    exit / quit            stop the session
    clear                  clear conversation history
    ingest [--reset]       re-run ingestion
"""

import os
import re
import sys

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from chunking import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, run_ingestion
from prompt import (
    SYSTEM_PROMPT,
    build_judgment_ratio_prompt,
    build_judgment_ratio_search_query,
    build_rag_prompt,
    is_judgment_ratio_query,
)

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
CHAT_MODEL        = "gpt-4o"
TOP_K             = 20
FINAL_K_JUDGMENT  = 6    # chunks sent to model for judgment ratio queries
FINAL_K_GENERAL   = 10   # more chunks for general questions — need broader context
MAX_HISTORY_TURNS = 10
PROBE_K           = 40


# ── ChromaDB ───────────────────────────────────────────────────────────────────

def get_retriever() -> chromadb.Collection:
    if not CHROMA_DIR.exists():
        print("  ChromaDB not found — ingesting first...\n")
        run_ingestion()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )
    try:
        col = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        print("  Collection not found — ingesting first...\n")
        run_ingestion()
        col = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    print(f"Connected — {col.count()} chunks indexed.\n")
    return col


# ── Case index ─────────────────────────────────────────────────────────────────

def build_case_index(col: chromadb.Collection) -> dict[str, str]:
    """
    Returns {source_filename: case_title} built from chunk_index==0 of each doc.
    Falls back to scanning chunk_index==1 if chunk 0 has no title (rare).
    """
    index: dict[str, str] = {}
    for ci in (0, 1):
        try:
            results = col.get(
                where={"chunk_index": {"$eq": ci}},
                include=["metadatas"],
                limit=500,
            )
            for meta in results.get("metadatas", []):
                source = meta.get("source", "")
                title  = meta.get("case_title", "").strip()
                if source and title and source not in index:
                    index[source] = title
        except Exception as e:
            print(f"  Case index warning (chunk {ci}): {e}")
    return index


# ── Query parsing ──────────────────────────────────────────────────────────────

def extract_petition_number_from_query(query: str) -> str | None:
    """
    Pull a petition / case number out of the user's query.
    e.g. 'judgment ratio for WP 23618' → '23618'
         'case No. 23618/2021'         → '23618'
    """
    patterns = [
        r"(?:W\.?P\.?|Writ Petition|Case|No\.?|Petition)\s*(?:No\.?)?\s*(\d{4,6}(?:/\d{4})?)",
        r"\b(\d{4,6}/\d{4})\b",   # bare 23618/2021
        r"\b(\d{5,6})\b",          # bare 5-6 digit number (petition numbers are usually 5 digits)
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ── Source resolution ──────────────────────────────────────────────────────────

def resolve_source_from_index(
    query: str,
    case_index: dict[str, str],
) -> tuple[str | None, str | None]:
    """
    Token-overlap match against known case titles.
    Returns (source, title) if ≥ 40% of non-noise title tokens appear in query.
    Threshold lowered to 40% to handle partial case names.
    """
    if not case_index:
        return None, None

    noise    = {"vs", "versus", "the", "of", "and", "ltd", "pvt",
                "mr", "ms", "mrs", "m", "s", "co", "on", "in"}
    q_tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()) - noise

    best_source, best_title, best_score = None, None, 0.0
    for source, title in case_index.items():
        t_tokens = set(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()) - noise
        if not q_tokens or not t_tokens:
            continue
        score = len(q_tokens & t_tokens) / len(t_tokens)
        if score > best_score:
            best_score, best_source, best_title = score, source, title

    return (best_source, best_title) if best_score >= 0.4 else (None, None)


def identify_source_by_embedding(
    query: str,
    col: chromadb.Collection,
) -> tuple[str | None, str | None]:
    """
    Fallback: run a broad similarity search and vote for the dominant source.
    Returns (source, title_or_empty).
    """
    n = min(PROBE_K, col.count())
    if n == 0:
        return None, None

    results = col.query(
        query_texts=[query],
        n_results=n,
        include=["metadatas", "distances"],
    )

    source_scores: dict[str, float] = {}
    source_titles: dict[str, str]   = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        src   = meta.get("source", "")
        title = meta.get("case_title", "")
        if src:
            source_scores[src] = source_scores.get(src, 0.0) + (1.0 - dist)
            if title and src not in source_titles:
                source_titles[src] = title

    if not source_scores:
        return None, None

    winner = max(source_scores, key=lambda s: source_scores[s])
    return winner, source_titles.get(winner, "")


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    col: chromadb.Collection,
    top_k: int = TOP_K,
    final_k: int = FINAL_K_GENERAL,
    is_judgment_query: bool = False,
    source_filter: str | None = None,
    petition_filter: str | None = None,
) -> tuple[str, list[dict], list[str]]:
    """
    Two-stage retrieval:
      Stage A — similarity search with hard source_filter (Fix 1/2).
      Stage B — if petition_filter is set, also fetch the last 20% of the
                locked document by position and keep only chunks whose
                all_cases field contains the target petition number (Fix 4).
                This guarantees the final-order chunk is included even if
                its embedding similarity is low.
    Results from both stages are merged, de-duplicated, re-ranked, and trimmed
    to final_k.
    """
    n = min(top_k, col.count())
    base_where: dict | None = {"source": {"$eq": source_filter}} if source_filter else None

    # Stage A: similarity search
    kw: dict = dict(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if base_where:
        kw["where"] = base_where

    res       = col.query(**kw)
    raw_docs  = res["documents"][0]
    raw_metas = res["metadatas"][0]
    raw_dists = res["distances"][0]

    # Stage B: position-based fetch for petition-specific final-order chunks
    extra_docs, extra_metas, extra_dists = [], [], []
    if source_filter and petition_filter and is_judgment_query:
        try:
            # Grab chunks from the last 20% of the document (position >= 0.8)
            end_results = col.get(
                where={
                    "$and": [
                        {"source":   {"$eq": source_filter}},
                        {"position": {"$gte": 0.8}},
                    ]
                },
                include=["documents", "metadatas"],
                limit=50,
            )
            seen_ids = set()
            for doc, meta in zip(
                end_results.get("documents", []),
                end_results.get("metadatas", []),
            ):
                all_cases = meta.get("all_cases", "")
                # Keep chunk if it mentions the target petition number
                petition_short = petition_filter.split("/")[0]  # strip /2021 suffix
                if petition_short in all_cases or petition_filter in all_cases:
                    uid = f"{meta.get('source')}-{meta.get('chunk_index')}"
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        extra_docs.append(doc)
                        extra_metas.append(meta)
                        extra_dists.append(0.05)  # treat as highly relevant
        except Exception as e:
            print(f"  Stage B fetch warning: {e}")

    # Merge and de-duplicate
    seen_uid  = set()
    all_items = []
    # Position boost only applies to judgment ratio queries (to surface final-order chunks).
    # For general queries it must be 0.0 — otherwise the final dismissal line
    # gets promoted to rank 1 even for "tell me about" questions.
    position_weight = 0.4 if is_judgment_query else 0.0

    for doc, meta, dist in (
        list(zip(raw_docs, raw_metas, raw_dists))
        + list(zip(extra_docs, extra_metas, extra_dists))
    ):
        uid = f"{meta.get('source')}-{meta.get('chunk_index')}"
        if uid in seen_uid:
            continue
        seen_uid.add(uid)
        similarity = 1.0 - dist
        position   = float(meta.get("position", 0.0))
        combined   = similarity + position_weight * position
        all_items.append((combined, similarity, doc, meta))

    all_items.sort(key=lambda x: x[0], reverse=True)
    all_items = all_items[:final_k]

    context_parts, documents, metadatas = [], [], []
    for rank, (_, sim, doc, meta) in enumerate(all_items, start=1):
        src      = meta.get("source", "?")
        ci       = meta.get("chunk_index", "?")
        total    = meta.get("total_chunks", "?")
        pos_pct  = round(float(meta.get("position", 0.0)) * 100, 1)
        rel      = round(sim * 100, 1)
        case_num = meta.get("case_number", "")
        all_c    = meta.get("all_cases", "")

        header = (
            f"[Rank {rank} | {src} | chunk {ci}/{total} | pos {pos_pct}% | "
            f"relevance {rel}%"
            + (f" | cases: {all_c}" if all_c else "")
            + "]"
        )
        context_parts.append(f"{header}\n{doc}")
        documents.append(doc)
        metadatas.append(meta)

    return "\n\n---\n\n".join(context_parts), metadatas, documents


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def chat_with_openai(client: OpenAI, user_prompt: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


# ── Display ────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("=" * 65)
    print("  Legal Document RAG — OpenAI + ChromaDB")
    print("=" * 65)
    print('  Try: "judgment ratio for M/S Elora Tobacco vs Union of India"')
    print("  Commands: exit | quit | clear | ingest | ingest --reset")
    print("=" * 65)
    print()


def print_sources(
    metadatas: list[dict],
    documents: list[str],
    locked_source: str | None,
    locked_title: str | None,
    petition_filter: str | None,
) -> None:
    if locked_source:
        label = f"{locked_title}  ({locked_source})" if locked_title else locked_source
        print(f"\n  Document locked : {label}")
    if petition_filter:
        print(f"  Petition filter : {petition_filter}")
    print("\n  Retrieved chunks:")
    for i, (meta, doc) in enumerate(zip(metadatas, documents), start=1):
        src     = meta.get("source", "?")
        ci      = meta.get("chunk_index", "?")
        total   = meta.get("total_chunks", "?")
        pos     = round(float(meta.get("position", 0.0)) * 100, 1)
        all_c   = meta.get("all_cases", "")
        c_str   = f"  cases: {all_c}" if all_c else ""
        print(f"\n  [{i}] {src}  chunk {ci}/{total}  pos {pos}%{c_str}")
        print(f"  {'-' * 60}")
        print(f"  {doc.strip()}")
    print()


# ── Chat loop ──────────────────────────────────────────────────────────────────

def run_chat() -> None:
    if not OPENAI_API_KEY:
        sys.exit("ERROR: OPENAI_API_KEY is not set.")

    client     = OpenAI(api_key=OPENAI_API_KEY)
    col        = get_retriever()

    print("  Building case index...")
    case_index = build_case_index(col)
    print(f"  {len(case_index)} document(s) in index.")
    for src, title in case_index.items():
        print(f"    {src}: {title}")
    print()

    print_banner()
    history: list[dict] = []

    while True:
        try:
            query = input("\nAsk: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if query.lower() == "clear":
            history.clear()
            print("  History cleared.\n")
            continue
        if query.lower().startswith("ingest"):
            run_ingestion(reset="--reset" in query.lower())
            col = get_retriever()
            case_index = build_case_index(col)
            print(f"  {len(case_index)} document(s) re-indexed.\n")
            continue

        # ── Fix 1+2: resolve which document the query is about ─────────────
        print("\n  Identifying document...")
        locked_source, locked_title = resolve_source_from_index(query, case_index)
        how = "case title match"

        if not locked_source:
            locked_source, locked_title = identify_source_by_embedding(query, col)
            how = "embedding vote"

        if locked_source:
            print(f"  Locked: {locked_source} — {locked_title}  [{how}]")
        else:
            print("  No lock — searching all documents.")

        # ── Fix 4: extract petition number from query ───────────────────────
        petition_filter = extract_petition_number_from_query(query)
        if petition_filter:
            print(f"  Petition filter: {petition_filter}")

        # ── Fix 5: enrich retrieval query for judgment mode ─────────────────
        judgment_mode   = is_judgment_ratio_query(query)
        retrieval_query = (
            build_judgment_ratio_search_query(query, locked_title)
            if judgment_mode
            else query
        )

        # ── Retrieve ────────────────────────────────────────────────────────
        context, metadatas, documents = retrieve_context(
            retrieval_query,
            col,
            final_k=FINAL_K_JUDGMENT if judgment_mode else FINAL_K_GENERAL,
            is_judgment_query=judgment_mode,
            source_filter=locked_source,
            petition_filter=petition_filter,
        )

        if not context.strip():
            print("\nAssistant: No relevant excerpts found.\n")
            continue

        # ── Build prompt with Fix 3 isolation guard ─────────────────────────
        if judgment_mode:
            print("  Judgment Ratio mode — generating answer...")
            user_prompt = build_judgment_ratio_prompt(
                context=context,
                question=query,
                case_title=locked_title,
                petition_number=petition_filter,
            )
        else:
            user_prompt = build_rag_prompt(
                context=context,
                question=query,
                case_title=locked_title,
                petition_number=petition_filter,
            )

        print("  Generating answer...\n")
        try:
            answer = chat_with_openai(client, user_prompt, history)
        except Exception as e:
            print(f"\n  OpenAI error: {e}\n")
            continue

        print_sources(metadatas, documents, locked_source, locked_title, petition_filter)
        print(f"Answer:\n{answer}")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    run_chat()