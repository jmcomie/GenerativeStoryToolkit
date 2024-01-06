import openai

import gstk.config as cfg


def get_openai_vectorization(input: str) -> str:
    """
    Get the OpenAI vectorization for a registration model.
    """
    return openai.Embedding.create(input=input, model=cfg.CHAT_GPT_EMBEDDING_MODEL)["data"][0]["embedding"]
