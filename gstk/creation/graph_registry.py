"""
Creation graph registry.
"""
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from gstk.graph.registry import ALL_NODES, EdgeCardinality, GraphRegistry, SystemEdgeType


class Role(StrEnum):
    system = "system"
    user = "user"
    function = "function"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None  # Only used for function role

    class Config:
        extra = "forbid"
        use_enum_values = True


class Labels(BaseModel):
    """
    Labels are names, tags or some other identifier that the user is creating.
    """

    desired_quantity: int = Field(description="The desired number of labels. Infer from prompt.")
    labels: list[str] = Field(description="The labels themselves, of the desired quantity.")


class ChatCompletionArguments(BaseModel):
    pass


class CreationNode(StrEnum):
    GROUP = "creation.group"
    MESSAGE = "creation.message"
    SELECTION = "creation.selection"
    LABELS = "creation.labels"
    ALL = "creation.*"


class CreationEdge(StrEnum):
    METADATA = "creation.metadata"
    CREATED_BY = "creation.created_by"


class GroupProperties(BaseModel):
    name: str


class SelectionData(BaseModel):
    """
    SelectionData is a set of node identifiers selected by the LLM in response to a user prompt
    and a textual reason that the particular node identifiers were selected.
    """

    node_identifiers: list[int] = Field(default=None, description="The identifiers of the selected nodes.")
    reason: str = Field(default=None, description="The reason(s) that the nodes were selected.")


GraphRegistry.register_node(CreationNode.GROUP, model=GroupProperties)

GraphRegistry.register_node(CreationNode.SELECTION, model=SelectionData)

GraphRegistry.register_node(CreationNode.MESSAGE, model=Message)

GraphRegistry.register_node(
    CreationNode.LABELS,
    model=Labels,
    system_message="You are tasked with interpreting some term for a identifier and interpreting it "
    + "as a list of labels according to the logic of the prompt provided.",
)

GraphRegistry.register_connection_types(
    CreationNode.GROUP, CreationNode.MESSAGE, [SystemEdgeType.REFERENCES, SystemEdgeType.CONTAINS]
)

GraphRegistry.register_connection_types(CreationNode.GROUP, CreationNode.GROUP, [SystemEdgeType.CONTAINS])

GraphRegistry.register_edge(
    CreationEdge.CREATED_BY, EdgeCardinality.MANY_TO_MANY, connection_data=[[ALL_NODES, CreationNode.MESSAGE]]
)
