"""Split text into overlapping token-bounded chunks."""
import tiktoken

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80

_enc = tiktoken.get_encoding("cl100k_base")


def chunk(text: str) -> list[str]:
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks
