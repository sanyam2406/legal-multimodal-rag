"""
prompt.py — Prompt templates for the Legal RAG pipeline.
"""

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a legal analyst. Answer only from the documents provided.
Never fabricate citations, case names, or legal principles.

Rules you must follow without exception:
- Be concise. Give only what is asked. No elaboration, no background, no reasoning steps.
- Output plain text only. No markdown, no bullet symbols, no asterisks, no headers, no bold, no lists.
- Synthesize your answer from whatever is present in the context, even if the context is partial.
  Only reply with "Not found in the provided documents" when the context contains zero relevant information.
- Cite the source file name inline at the end of your answer in parentheses, nothing more.
- If multiple cases or petition numbers appear in the context, answer only for the case or petition number explicitly stated in the question."""


# ── Case isolation guard ───────────────────────────────────────────────────────

def _case_guard(case_title: str | None, petition_number: str | None) -> str:
    """
    Generates a hard isolation instruction injected at the top of every prompt.
    Handles both full case title and specific petition number (e.g. 23618 vs 23624).
    """
    parts = []
    if case_title:
        parts.append(f"Answer only for the case: {case_title}.")
    if petition_number:
        # Strip year suffix so "23618" matches "23618/2021" in context
        short = petition_number.split("/")[0]
        parts.append(
            f"If multiple petition numbers appear, answer only for petition No. {petition_number} "
            f"(or {short}). Ignore outcomes, facts, and reasoning from any other petition number."
        )
    if parts:
        return "IMPORTANT: " + " ".join(parts) + " Do not mix cases.\n\n"
    return ""


# ── RAG QA prompt ──────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """{case_guard}Answer the question below using only the retrieved excerpts. \
Do not use external knowledge. Do not repeat the question. Output plain text only, no markdown.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def build_rag_prompt(
    context: str,
    question: str,
    case_title: str | None = None,
    petition_number: str | None = None,
) -> str:
    return RAG_PROMPT_TEMPLATE.format(
        case_guard=_case_guard(case_title, petition_number),
        context=context,
        question=question,
    )


# ── Judgment Ratio prompt ──────────────────────────────────────────────────────
JUDGMENT_RATIO_PROMPT_TEMPLATE = """{case_guard}The context below contains excerpts from a court judgment. \
State the ratio decidendi — the binding legal principle that drove the decision — in one or two sentences. \
Plain text only. No labels, no markdown, no preamble, no case name.

How to find the ratio:
- Look for the court's own legal reasoning: what legal rule or principle did the court apply to reach its conclusion?
- The reasoning is usually in chunks with phrases like "cannot compel", "liable to be struck down", \
"without jurisdiction", "held that", "in view of the above".
- The outcome (allowed / dismissed) tells you the result, not the ratio. The ratio is WHY.
- If the document covers multiple petitions, use the outcome of the petition that was ALLOWED \
as the primary ratio unless the question specifies otherwise.
- You MUST synthesize a ratio from whatever legal reasoning appears in the context. \
Only reply "Not determinable from available documents" if the context contains no legal reasoning at all — \
not even a single sentence explaining why the court ruled as it did.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def build_judgment_ratio_prompt(
    context: str,
    question: str,
    case_title: str | None = None,
    petition_number: str | None = None,
) -> str:
    return JUDGMENT_RATIO_PROMPT_TEMPLATE.format(
        case_guard=_case_guard(case_title, petition_number),
        context=context,
        question=question,
    )


# ── Judgment ratio retrieval query ─────────────────────────────────────────────

def build_judgment_ratio_search_query(
    user_query: str,
    case_title: str | None,
) -> str:
    """
    Fix 5: enrich retrieval query with case name + final-order legal language.
    Anchors the embedding to the specific case rather than generic legal boilerplate.
    """
    legal_terms = (
        "final decision petition allowed dismissed final order of court "
        "writ allowed disposed held that accordingly ordered judgment"
    )
    if case_title:
        return f"{case_title} {legal_terms}"
    return f"{user_query} {legal_terms}"


# ── Trigger detection ──────────────────────────────────────────────────────────

JUDGMENT_RATIO_TRIGGERS = [
    "judgment ratio",
    "judgement ratio",
    "ratio decidendi",
    "ratio of judgment",
    "ratio of judgement",
    "why was the decision",
    "why was the judgment",
    "why was the judgement",
    "reason for the decision",
    "reasoning behind the decision",
    "why the decision was taken",
    "why the judgment was given",
]


def is_judgment_ratio_query(query: str) -> bool:
    q = query.lower()
    return any(trigger in q for trigger in JUDGMENT_RATIO_TRIGGERS)