"""
Creation graph registry.
"""
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from gstk.graph.registry import ALL_NODES, EdgeCardinality, EdgeRegistry, NodeRegistry
from gstk.graph.system_graph_registry import SystemEdgeRegistry, SystemEdgeType, SystemNodeRegistry

CreationNodeRegistry: NodeRegistry = SystemNodeRegistry.clone()
CreationEdgeRegistry: EdgeRegistry = SystemEdgeRegistry.clone()


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


class ChatCompletionArguments(BaseModel):
    pass


class CreationNode(StrEnum):
    group = "creation.group"
    message = "creation.message"
    group_collection_data = "creation.group_collection_data"
    group_summary_data = "creation.group_summary_data"
    selection = "creation.selection"
    ALL = "creation.*"


class CreationEdge(StrEnum):
    metadata = "creation.metadata"
    created_by = "creation.created_by"


class GroupProperties(BaseModel):
    name: str


class GroupCollectionData(BaseModel):
    pass


class GroupSummaryData(BaseModel):
    pass


class SelectionData(BaseModel):
    """
    SelectionData is a set of node identifiers selected by the LLM in response to a user prompt
    and a textual reason that the particular node identifiers were selected.
    """

    node_identifiers: list[int] = Field(default=None, description="The identifiers of the selected nodes.")
    reason: str = Field(default=None, description="The reason(s) that the nodes were selected.")


CreationNodeRegistry.register_node(CreationNode.group, model=GroupProperties)

CreationNodeRegistry.register_node(CreationNode.selection, model=SelectionData)

CreationNodeRegistry.register_node(CreationNode.message, model=Message)

CreationNodeRegistry.register_node(CreationNode.group_collection_data, model=GroupCollectionData)

CreationNodeRegistry.register_node(CreationNode.group_summary_data, model=GroupSummaryData)

CreationEdgeRegistry.register_connection_types(
    CreationNode.group, CreationNode.message, [SystemEdgeType.references, SystemEdgeType.contains]
)

CreationEdgeRegistry.register_connection_types(CreationNode.group, CreationNode.group, [SystemEdgeType.contains])

CreationEdgeRegistry.register_edge(
    CreationEdge.metadata,
    EdgeCardinality.ONE_TO_MANY,
    connection_data=[
        [CreationNode.group, CreationNode.group_collection_data],
        [CreationNode.group, CreationNode.group_summary_data],
    ],
)

CreationEdgeRegistry.register_edge(
    CreationEdge.created_by, EdgeCardinality.MANY_TO_MANY, connection_data=[[ALL_NODES, CreationNode.message]]
)
