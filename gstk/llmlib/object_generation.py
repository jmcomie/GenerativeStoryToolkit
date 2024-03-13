

import json
from pydantic import BaseModel
from gstk.config import ChatGPTModel, CHAT_GPT_MODEL
from gstk.creation.graph_registry import Message
from gstk.graph.registry import GraphRegistry, NodeTypeData
from gstk.llmlib.async_openai import get_chat_completion_response, get_function_tool


async def get_chat_completion_object_response(
        node_type: str, messages: list[Message], model: ChatGPTModel = CHAT_GPT_MODEL) -> BaseModel:
    """
    Sends the message list to OpenAI and returns the response as an instance of the
    configured type for the node_type. This serves as the primary bridge between
    the GSTK creation and graph pattern and OpenAI chat completion.
    """
    node_type_data: NodeTypeData = GraphRegistry.get_node_type_data(node_type)
    open_ai_normalized_node_type: str = node_type.replace(".", "_")
    response = await get_chat_completion_response(
        messages, tools=[get_function_tool(open_ai_normalized_node_type, node_type_data.model)]
    )
    instance: BaseModel = node_type_data.model(
        **json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    )
    return instance



