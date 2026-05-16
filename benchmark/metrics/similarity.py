import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return _model


def batch_similarity(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Cosine similarity between paired text lists. Returns array of shape (N,)."""
    model = _get_model()
    embs_a = model.encode(texts_a, convert_to_tensor=True, normalize_embeddings=True)
    embs_b = model.encode(texts_b, convert_to_tensor=True, normalize_embeddings=True)
    return (embs_a * embs_b).sum(dim=1).cpu().numpy()


def token_count(texts: list[str], tokenizer) -> list[int]:
    return [len(tokenizer.encode(t)) for t in texts]
