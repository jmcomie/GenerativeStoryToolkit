from enum import StrEnum

class OpenAIModelName(StrEnum):
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4_1106_VISION_PREVIEW = "gpt-4-1106-vision-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_0613 = "gpt-4-0613"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

CHAT_GPT_MODEL: OpenAIModelName = OpenAIModelName.GPT_4_0125_PREVIEW
# https://platform.openai.com/docs/guides/embeddings/embedding-models
CHAT_GPT_EMBEDDING_MODEL: OpenAIModelName = OpenAIModelName.TEXT_EMBEDDING_ADA_002
CHAT_GPT_TEMPERATURE: int = 0.1
VECTOR_COSINE_DUPLICATE_THRESHOLD: float = 0.9
# https://platform.openai.com/docs/models/overview



