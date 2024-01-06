from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

import gstk.config as cfg


class Role(StrEnum):
    system = "system"
    user = "user"
    function = "function"
    assistant = "assistant"


class GFTKMetadata(BaseModel):
    output_identifier: Optional[str] = None
    is_creation_output: Optional[bool] = False


class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None  # Only used for function role

    class Config:
        extra = "forbid"
        use_enum_values = True


class ChatCompletionArguments(BaseModel):
    id: str
    base_messages: list[Message]
    temperature: float = cfg.CHAT_GPT_TEMPERATURE
    model: str = cfg.CHAT_GPT_MODEL
    function_call: str | dict[str, str] = "auto"
    send_function_response_context: bool = False
    function_call_limit: Optional[int] = 1
    messages: list[Message] = []

    class Config:
        extra = "forbid"
        use_enum_values = True


class EmbeddingVector(BaseModel):
    vector: list[float]
