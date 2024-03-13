from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

import gstk.config as cfg


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
