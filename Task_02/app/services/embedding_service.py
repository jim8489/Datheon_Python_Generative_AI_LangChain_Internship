from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and cache the embedding model.

    The model runs locally, so document embeddings do not
    require a paid embedding API.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.
    """

    if not texts:
        raise ValueError("Cannot generate embeddings for an empty list.")

    model = get_embedding_model()

    return model.embed_documents(texts)