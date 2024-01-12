"""
Establishes the core system graph structures. The node and edge registries can be
cloned and modified to create custom graph structures.
"""

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from gstk.graph.registry import ALL_NODES, EdgeCardinality, EdgeRegistry, NodeRegistry

SystemNodeRegistry = NodeRegistry()
SystemEdgeRegistry = EdgeRegistry()


class ProjectProperties(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    supplemental_data: Optional[dict] = None


class MediaProperties(BaseModel):
    path: str
    name: Optional[str] = None
    media_vector: Optional[list[float]] = None


class SystemNodeType(StrEnum):
    project = "system.project"
    media = "system.media"
    ALL = "system.*"


class SystemEdgeType(StrEnum):
    clone = "system.clone"
    contains = "system.contains"
    references = "system.references"


SystemNodeRegistry.register_node(SystemNodeType.project, model=ProjectProperties, instance_limit=1)

SystemNodeRegistry.register_node(SystemNodeType.media, model=MediaProperties)

SystemEdgeRegistry.register_edge(
    SystemEdgeType.references,
    edge_cardinality=EdgeCardinality.MANY_TO_MANY,
)

SystemEdgeRegistry.register_edge(
    SystemEdgeType.clone, edge_cardinality=EdgeCardinality.ONE_TO_MANY, connection_data=[[ALL_NODES, ALL_NODES]]
)

SystemEdgeRegistry.register_connection_type(ALL_NODES, SystemNodeType.media, SystemEdgeType.references)

SystemEdgeRegistry.register_edge(SystemEdgeType.contains, EdgeCardinality.ONE_TO_MANY)
