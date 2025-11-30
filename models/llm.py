# models/llm.py
from typing import Dict, List
import google.generativeai as genai

from config.config import settings
from models.embeddings import retrieve_relevant_chunks

# Configure Gemini
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


def build_prompt(
    query: str,
    context: str,
    chat_history: List[Dict[str, str]],
) -> str:
    history_lines = []
    for m in chat_history[-20:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {m['content']}")

    prompt = f"""
You are an AI assistant for a hotel that helps guests with questions
and room bookings. Always be polite and concise.

You have the following hotel information from PDFs:
--------------------
{context}
--------------------

Conversation so far:
{chr(10).join(history_lines)}

Now the user asks: "{query}"

1. If the answer is clearly in the context, use it.
2. If it's not in the context, answer from general hotel knowledge
   and mention that it is general knowledge.
3. If you don't know, say you are not sure.

Assistant:
"""
    return prompt.strip()


def answer_with_rag(
    query: str,
    vector_store: Dict,
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Returns a string answer. Behavior:
      - If no API key: return TF-IDF retrieved chunks (if available) or a helpful message.
      - If API key present: try Gemini. On any error, return TF-IDF chunks (if available)
        plus the Gemini error text so the UI shows the cause.
    """
    # Get TF-IDF context (may be empty)
    try:
        chunks = retrieve_relevant_chunks(query, vector_store, top_k=3)
    except Exception:
        chunks = []

    context = "\n\n".join(chunks) if chunks else "No hotel PDF context available."
    prompt = build_prompt(query, context, chat_history)

    # No API key: use deterministic TF-IDF fallback
    if not settings.GEMINI_API_KEY:
        if chunks:
            return "\n\n---\n\n".join([c[:1200] for c in chunks])
        return "Gemini API key not configured and no brochure context found. Please set GEMINI_API_KEY or upload a brochure."

    # Try to call Gemini, but handle errors gracefully
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception as e:
        # Provide fallback content + helpful debug note
        err_msg = f"[Gemini error] {type(e).__name__}: {e}"
        if chunks:
            fallback = "\n\n---\n\n".join([c[:1200] for c in chunks])
            return f"{fallback}\n\n\n(Notice: LLM call failed — {err_msg})"
        else:
            return f"LLM call failed: {err_msg}\n\nAlso, no PDF context was found to fall back to."

