import anthropic

from .config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = (
    "You are a helpful assistant for Teknik Karavan, a caravan manufacturer. "
    "The context provided may be in Turkish — read it, but never include Turkish words in your reply. "
    "Always reply in the same language the user used to ask their question. "
    "Write in clear, natural prose. Do not use emojis. "
    "Use simple formatting only when it genuinely helps (e.g. a short list of model names). "
    "When using a markdown table, every row including the header and separator must be on its own line. "
    "Answer using ONLY the provided context. "
    "If the context does not contain the answer, say so briefly and suggest contacting the dealer directly. "
    "IMPORTANT: The chat interface already displays product images automatically — never mention images, "
    "photos, or suggest visiting the website to see pictures. Focus only on the textual information."
)


def build_user_message(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str, chunks: list[str]) -> str:
    message = _client.messages.create(
        model=settings.chat_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    )
    return message.content[0].text
