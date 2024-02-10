import copy
import re
from enum import Enum, StrEnum
from functools import cache
from typing import Iterator, Optional, Type

from pydantic import BaseModel

ALL_NODES: str = "*"
NodeTypeInPolicyRegex: re.Pattern = re.compile(r"(([a-z_]+)\.([a-z_]+|\*))$|(\*)")
NodeOrEdgeTypeRegex: re.Pattern = re.compile(r"([a-z_]+)\.([a-z_]+)")


class NodeTypeData(BaseModel):
    instance_limit: Optional[int] = None
    write_allowed: bool = True
    system_directive: Optional[str] = None
    model: Type[BaseModel] = BaseModel

    class Config:
        extra = "forbid"
        use_enum_values = True


class EdgeCardinality(Enum):
    ONE_TO_ONE = 0
    ONE_TO_MANY = 1
    MANY_TO_ONE = 2
    MANY_TO_MANY = 3


class EdgeTypeData(BaseModel):
    edge_cardinality: EdgeCardinality
    write_allowed: bool = True
    connection_data: set[tuple[str, str]]

    class Config:
        extra = "forbid"
        use_enum_values = True


def check_node_or_edge_type(node_or_edge_type: str):
    if not NodeOrEdgeTypeRegex.match(node_or_edge_type):
        raise ValueError(f"Invalid node or edge type: {node_or_edge_type}")


def check_node_type_in_connection_policy(type_in_policy: str):
    if not NodeTypeInPolicyRegex.match(type_in_policy):
        raise ValueError(f"Invalid node type in connection policy: {type_in_policy}")


@cache
def node_type_matches_type_in_policy(node_type: str, type_in_policy: str):
    check_node_or_edge_type(node_type)
    check_node_type_in_connection_policy(type_in_policy)
    if type_in_policy == ALL_NODES:
        return True
    elif type_in_policy == node_type:
        return True
    elif type_in_policy.endswith(ALL_NODES):
        return node_type.startswith(type_in_policy[:-1])
    else:
        return False


@cache
def node_type_matches_type_in_policy_list(node_type: str | list[str], type_in_policy_list: list[str]):
    return any([node_type_matches_type_in_policy(node_type, type_in_policy) for type_in_policy in type_in_policy_list])


class GraphRegistry:
    node_type_map: dict[str, NodeTypeData] = {}
    edge_type_map: dict[str, EdgeTypeData] = {}

    @classmethod
    def node_type(cls, node_type, instance_limit: Optional[int] = None):
        if node_type in cls.registry:
            raise ValueError(f"Node type {node_type} already registered")

        def decorator(model: type):
            system_message: Optional[str] = getattr(model, "_system_message", None)
            cls.register_node(node_type, model, instance_limit=instance_limit, system_message=system_message)
            return model

        return decorator

    def get_node_types(cls, model_type: type) -> Iterator[str]:
        for node_type, node_type_data in cls.node_type_map.items():
            if node_type_data.model == model_type:
                yield node_type
        raise ValueError(f"Model {model_type} not registered.")

    def get_model(cls, node_type) -> type:
        return cls.registry[node_type]

    def register_node(
        self,
        node_type: str,
        model: Type[BaseModel],
        instance_limit: Optional[int] = None,
        system_message: Optional[str] = None,
    ):
        if node_type in self.node_type_map:
            raise Exception(f"Node type {node_type} already registered.")
        self.node_type_map[node_type] = NodeTypeData(
            instance_limit=instance_limit, model=model, system_directive=system_message
        )

    def get_node_type_data(self, node_type: str) -> NodeTypeData:
        if node_type not in self.node_type_map:
            raise Exception(f"Node type {node_type} not registered.")
        return self.node_type_map[node_type]

    def register_edge(
        self,
        edge_type: str,
        edge_cardinality: EdgeCardinality,
        system_message: Optional[str] = None,
        connection_data: Optional[list[tuple[str, str]]] = None,
    ):
        check_node_or_edge_type(edge_type)
        if edge_type in self.edge_type_map:
            raise Exception(f"Edge type {edge_type} already registered.")
        if connection_data is None:
            connection_data = []
        self.edge_type_map[edge_type] = EdgeTypeData(edge_cardinality=edge_cardinality, connection_data=connection_data)

    def register_connection_type(self, from_node_type: str, to_node_type: str, edge_type: str):
        # If we have cardinality here, what does it look like?
        if not self.is_registered(edge_type):
            raise Exception(f"Edge type {edge_type} not registered.")
        self.edge_type_map[edge_type].connection_data.add((from_node_type, to_node_type))

    def register_connection_types(self, from_node_type: str, to_node_type: str, edge_type_list: list[str]):
        for edge_type in edge_type_list:
            self.register_connection_type(from_node_type, to_node_type, edge_type)

    def get_edge_type_data(self, edge_type: str) -> EdgeTypeData:
        if edge_type not in self.edge_type_map:
            raise Exception(f"Edge type {edge_type} not registered.")
        return self.edge_type_map[edge_type]

    def is_registered_connection(self, from_node_type: str, to_node_type: str, edge_type: str) -> bool:
        for from_node_type_in_policy, to_node_type_in_policy in self.edge_type_map[edge_type].connection_data:
            if node_type_matches_type_in_policy(
                from_node_type, from_node_type_in_policy
            ) and node_type_matches_type_in_policy(to_node_type, to_node_type_in_policy):
                return True
        return False


class SystemNodeType(StrEnum):
    project = "system.project"
    media = "system.media"
    ALL = "system.*"


class SystemEdgeType(StrEnum):
    clone = "system.clone"
    contains = "system.contains"
    references = "system.references"


@GraphRegistry.node_type(SystemNodeType.project.value, instance_limit=1)
class ProjectProperties(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    supplemental_data: Optional[dict] = None


@GraphRegistry.node_type(SystemNodeType.media.value)
class MediaProperties(BaseModel):
    path: str
    name: Optional[str] = None
    media_vector: Optional[list[float]] = None


GraphRegistry.register_edge(
    SystemEdgeType.references,
    edge_cardinality=EdgeCardinality.MANY_TO_MANY,
)
GraphRegistry.register_edge(
    SystemEdgeType.clone, edge_cardinality=EdgeCardinality.ONE_TO_MANY, connection_data=[[ALL_NODES, ALL_NODES]]
)
GraphRegistry.register_edge(SystemEdgeType.contains, EdgeCardinality.ONE_TO_MANY)

# Any node can contain or reference any other node.
GraphRegistry.register_connection_types(ALL_NODES, ALL_NODES, [SystemEdgeType.contains, SystemEdgeType.references])
