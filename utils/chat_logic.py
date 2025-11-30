# utils/chat_logic.py
from typing import Dict, Tuple
import re

from models.llm import answer_with_rag

# Keep the same question words heuristic as in app.py
QUESTION_WORDS = ("what", "which", "who", "when", "where", "why", "how",
                  "do", "does", "is", "are", "can")

INFO_RE = re.compile(
    r"\b(what|which|when|where|why|how|tell|show|describe|list|types|kinds|rooms|amenities|facilities|how many)\b",
    re.IGNORECASE
)


def looks_like_info_question(text: str) -> bool:
    """
    Heuristic: question mark OR starts with question word OR starts with 'tell me', 'show', etc.
    Matches the heuristic used in app.py so app.py and chat_logic agree.
    """
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    if "?" in t:
        return True
    if any(t.startswith(w + " ") for w in QUESTION_WORDS):
        return True
    if t.startswith(("tell me", "show", "describe", "list", "give me")):
        return True
    if "types of" in t or "amenities" in t or "facilities" in t:
        return True
    return False


def handle_user_message(user_message: str, session_state: Dict) -> Tuple[str, Dict]:
    """
    Minimal router used by app.py.
    - If message looks like info question -> call RAG/LLM and return reply.
    - Otherwise also call RAG/LLM (app.py handles booking flow separately).
    Returns (reply, updated_session_state).
    """
    # keep session_state untouched except possibly appending assistant message
    messages = session_state.get("messages", [])
    vector_store = session_state.get("vector_store", {})

    # If it's clearly an info question prefer RAG
    try:
        reply = answer_with_rag(
            query=user_message,
            vector_store=vector_store,
            chat_history=messages,
        )
    except Exception as e:
        # safe fallback
        reply = f"Sorry — I had an internal error while searching the brochure: {type(e).__name__}: {e}"

    # Append assistant reply into session_state.messages if not duplicate.
    if not session_state.get("messages") or session_state["messages"][-1].get("content") != reply:
        session_state.setdefault("messages", []).append({"role": "assistant", "content": reply})

    return reply, session_state
