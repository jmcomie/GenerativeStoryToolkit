CHAT_GPT_MODEL: str = "gpt-3.5-turbo-16k"
# https://platform.openai.com/docs/guides/embeddings/embedding-models
CHAT_GPT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
CHAT_GPT_TEMPERATURE: int = 0.1
VECTOR_COSINE_DUPLICATE_THRESHOLD: float = 0.9
# https://platform.openai.com/docs/models/overview
CHAT_GPT_MODEL_TOKEN_LIMIT_DICT: dict[str, int] = {
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
}
CONTEXT_TRUNCATE_THRESHOLD: float = 0.8
CONTEXT_TRUNCATE_SEQUENCE: list[int] = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50]
IMAGE_DATA_FILENAME = "image_data"
