from enum import StrEnum
from functools import reduce
import operator
from typing import Optional

from openai import ChatCompletion
from pydantic import BaseModel

from gstk.config import CHAT_GPT_MODEL, CHAT_GPT_TEMPERATURE, OpenAIModelName


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    FUNCTION = "function"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None  # Only used for function role

    class Config:
        extra = "forbid"
        use_enum_values = True

    def __str__(self):
        return f"{str(self.role).upper()} MESSAGE:\n{'-' * len(str(self.role + ' MESSAGE:'))}\n{self.content}"


class ChatCompletionArguments(BaseModel):
    id: str
    base_messages: list[Message]
    temperature: float = CHAT_GPT_TEMPERATURE
    model: str = CHAT_GPT_MODEL
    function_call: str | dict[str, str] = "auto"
    send_function_response_context: bool = False
    function_call_limit: Optional[int] = 1
    messages: list[Message] = []

    class Config:
        extra = "forbid"
        use_enum_values = True


class EmbeddingVector(BaseModel):
    vector: list[float]

# https://openai.com/pricing

    # below includes model name, training cost, input token cost, and output token cost

class OpenAIModel(BaseModel):
    name: OpenAIModelName
    training_cost_per_m_tokens_usd: Optional[float] = None
    input_cost_per_m_tokens_usd: Optional[float] = None
    output_cost_per_m_tokens_usd: Optional[float] = None
    max_context_length: Optional[int] = None
    context_window: Optional[int] = None

# Fine-tuning link to be reviewed:
# https://platform.openai.com/docs/guides/fine-tuning/use-a-fine-tuned-model



# sources:
# https://openai.com/pricing
# https://platform.openai.com/docs/models/overview

MODELS: list[OpenAIModel] = [
    OpenAIModel(
        name=OpenAIModelName.GPT_4_0125_PREVIEW,
        input_cost_per_m_tokens_usd=10,
        output_cost_per_m_tokens_usd=30,
        context_window=128_000
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_4_1106_PREVIEW,
        input_cost_per_m_tokens_usd=10,
        output_cost_per_m_tokens_usd=30,
        context_window=128_000
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_4_1106_VISION_PREVIEW,
        input_cost_per_m_tokens_usd=10,
        output_cost_per_m_tokens_usd=30,
        context_window=128_000
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_4,
        input_cost_per_m_tokens_usd=30,
        output_cost_per_m_tokens_usd=60,
        context_window=8192
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_4_32K,
        input_cost_per_m_tokens_usd=60,
        output_cost_per_m_tokens_usd=120,
        context_window=32_768
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_3_5_TURBO_0125,
        input_cost_per_m_tokens_usd=0.50,
        output_cost_per_m_tokens_usd=1.50,
        context_window=16_385
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_3_5_TURBO_INSTRUCT,
        input_cost_per_m_tokens_usd=1.50,
        output_cost_per_m_tokens_usd=2.00,
        context_window=4096
    ),
    OpenAIModel(
        name=OpenAIModelName.GPT_4_0613,
        input_cost_per_m_tokens_usd=30,
        output_cost_per_m_tokens_usd=60,
        context_window=8192
    ),
    OpenAIModel(
        name=OpenAIModelName.TEXT_EMBEDDING_ADA_002
    )
]

class TokenCosts(BaseModel):
    input_cost: float
    output_cost: float

    class Config:
        extra = "forbid"

    def __str__(self):
        return f"Input Cost: ${self.input_cost:.2f}\nOutput Cost: ${self.output_cost:.2f}\nTotal Cost: ${self.input_cost + self.output_cost:.2f}"

    def __add__(self, other: "TokenCosts"):
        return TokenCosts(input_cost=self.input_cost + other.input_cost, output_cost=self.output_cost + other.output_cost)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    class Config:
        extra = "forbid"

    def __str__(self):
        return f"Input Tokens: {self.input_tokens}\nOutput Tokens: {self.output_tokens}"

    def __add__(self, other: "TokenUsage"):
        return TokenUsage(input_tokens=self.input_tokens + other.input_tokens, output_tokens=self.output_tokens + other.output_tokens)

def get_model_by_name(name: str|OpenAIModelName) -> OpenAIModel:
    for model in MODELS:
        if model.name == name:
            return model
    raise ValueError(f"Model {name} not found")


def get_token_dollar_cost(model_name: str|OpenAIModelName, token_usage: TokenUsage) -> TokenCosts:
    model = get_model_by_name(model_name)
    input_cost = (token_usage.input_tokens / 1_000_000) * model.input_cost_per_m_tokens_usd
    output_cost = (token_usage.output_tokens / 1_000_000) * model.output_cost_per_m_tokens_usd
    return TokenCosts(input_cost=input_cost, output_cost=output_cost)


def get_chat_completions_cost(chat_completions: list[ChatCompletion|dict]) -> TokenCosts:
    model_usage_dict: dict[str, TokenUsage] = {}
    for chat_completion in chat_completions:
        if isinstance(chat_completion, dict):
            model = chat_completion["model"]
            token_usage: TokenUsage = TokenUsage(input_tokens=chat_completion["usage"]["prompt_tokens"], output_tokens=chat_completion["usage"]["completion_tokens"])
        else:
            print(chat_completion)
            model = chat_completion.model
            token_usage: TokenUsage = TokenUsage(input_tokens=chat_completion.usage.prompt_tokens, output_tokens=chat_completion.usage.completion_tokens)
        if model not in model_usage_dict:
            model_usage_dict[model] = TokenUsage(input_tokens=0, output_tokens=0)
        model_usage_dict[model] += token_usage
    return reduce(operator.add,
        [get_token_dollar_cost(model, token_usage) for model, token_usage in model_usage_dict.items()])


#CHAT_GPT_MODEL: ChatGPTModel = ChatGPTModel.GPT_4_TURBO_0125
# https://platform.openai.com/docs/models/overview
# https://platform.openai.com/docs/guides/embeddings/embedding-models
CHAT_GPT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
CHAT_GPT_TEMPERATURE: int = 0.1
VECTOR_COSINE_DUPLICATE_THRESHOLD: float = 0.9

