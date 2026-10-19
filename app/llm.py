import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from . import config

# Load environment variables from the project's .env file.
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
_client = None


def client():
    """Create and reuse the Gemini client."""
    global _client

    if _client is None:
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not api_key:
            raise RuntimeError(
                "Gemini API key is missing. Add GEMINI_API_KEY "
                "to your project's .env file."
            )

        _client = genai.Client(api_key=api_key)

    return _client


SYSTEM = """You are a real-time IT troubleshooting voice assistant.

Answer ONLY from the numbered context passages.
If they do not contain the answer, say so briefly and suggest
what information to gather or when to escalate to IT support.
Never invent troubleshooting steps.

Style:
- Lead with the most likely fix.
- Use short, numbered steps and plain language.
- Avoid tables and long paragraphs.
- Keep answers under approximately 150 words unless asked for more.
- Cite sources inline as [1], [2], immediately after supported steps.
- If the problem is ambiguous, ask ONE clarifying question
  after giving the best first step.
"""


def build_context(chunks):
    """Format retrieved document chunks for grounded generation."""
    passages = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}

        source = metadata.get("source", "Unknown source")
        section = metadata.get("section", "")
        category = metadata.get("category", "")

        passages.append(
            f"[{i}] (source: {source} | section: {section} "
            f"| category: {category})\n{chunk.get('text', '')}"
        )

    return "\n\n".join(passages)


def _content(role: str, text: str) -> types.Content:
    """Convert conversation messages into Gemini content."""
    return types.Content(
        role="model" if role == "assistant" else "user",
        parts=[types.Part(text=text)],
    )


async def rewrite_query(query: str, history: list) -> str:
    """Resolve follow-ups into standalone IT troubleshooting queries."""
    if not history:
        return query

    conversation = "\n".join(
        f"{role}: {text}" for role, text in history[-4:]
    )

    response = await client().aio.models.generate_content(
        model=config.REWRITE_MODEL,
        contents=f"{conversation}\nuser: {query}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "Rewrite the user's last message as a standalone IT "
                "troubleshooting search query using the conversation "
                "for context. Output only the query."
            ),
            max_output_tokens=60,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return (response.text or "").strip() or query


async def stream_answer(query: str, chunks, history: list):
    """Stream a grounded answer using retrieved knowledge-base passages."""
    contents = [
        _content(role, text)
        for role, text in history[-config.MAX_HISTORY_TURNS:]
    ]

    context = (
        build_context(chunks)
        if chunks
        else "(no relevant passages found)"
    )

    contents.append(
        _content(
            "user",
            f"Context passages:\n{context}\n\nQuestion: {query}",
        )
    )

    stream = await client().aio.models.generate_content_stream(
        model=config.LLM_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=600,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    async for chunk in stream:
        if chunk.text:
            yield chunk.text
