import copy
import re
from enum import Enum
from functools import cache
from typing import Optional, Type

from pydantic import BaseModel

ALL_NODES: str = "*"
NodeTypeInPolicyRegex: re.Pattern = re.compile(r"((['a-z']+)\.(['a-z']+|\*))$|(\*)")
NodeOrEdgeTypeRegex: re.Pattern = re.compile(r"(['a-z']+)\.(['a-z']+)")


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
    print("in node_type_matches_type_in_policy_list")
    print(f"node_type_matches_type_in_policy_list: {node_type}, {type_in_policy_list}")
    return any([node_type_matches_type_in_policy(node_type, type_in_policy) for type_in_policy in type_in_policy_list])


class EdgeRegistry:
    def __init__(self, edge_type_map: Optional[dict[str, EdgeTypeData]] = None):
        self._edge_type_map = edge_type_map or dict()

    def register_edge(
        self,
        edge_type: str,
        edge_cardinality: EdgeCardinality,
        system_message: Optional[str] = None,
        connection_data: Optional[list[tuple[str, str]]] = None,
    ):
        check_node_or_edge_type(edge_type)
        if edge_type in self._edge_type_map:
            raise Exception(f"Edge type {edge_type} already registered.")
        if connection_data is None:
            connection_data = []
        self._edge_type_map[edge_type] = EdgeTypeData(
            edge_cardinality=edge_cardinality, connection_data=connection_data
        )

    def deregister_edge(self, edge_type: str):
        if edge_type not in self._edge_type_map:
            raise Exception(f"Edge type {edge_type} not registered.")

    def register_connection_type(self, from_node_type: str, to_node_type: str, edge_type: str):
        # If we have cardinality here, what does it look like?

        if not self.is_registered(edge_type):
            raise Exception(f"Edge type {edge_type} not registered.")
        self._edge_type_map[edge_type].connection_data.add((from_node_type, to_node_type))

    def register_connection_types(self, from_node_type: str, to_node_type: str, edge_type_list: list[str]):
        for edge_type in edge_type_list:
            self.register_connection_type(from_node_type, to_node_type, edge_type)

    def deregister_connection_types(self, from_node_type, to_node_type, edge_type):
        if not self.is_registered(edge_type):
            raise Exception(f"Connection types {from_node_type}, {to_node_type}, {edge_type} not registered.")
        self._edge_type_map[edge_type].connection_data.remove((from_node_type, to_node_type))

    def disable_edge(self, edge_type: str):
        self._edge_type_map[edge_type].write_allowed = False

    def enable_edge(self, edge_type: str):
        self._edge_type_map[edge_type].write_allowed = True

    def is_enabled(self, edge_type: str) -> bool:
        return self._edge_type_map[edge_type].write_allowed

    def is_registered(self, edge_type: str) -> bool:
        return edge_type in self._edge_type_map

    def get_edge_type_data(self, edge_type: str) -> EdgeTypeData:
        if edge_type not in self._edge_type_map:
            raise Exception(f"Edge type {edge_type} not registered.")
        return self._edge_type_map[edge_type]

    def clone(self) -> Type["EdgeRegistry"]:
        return EdgeRegistry(copy.deepcopy(self._edge_type_map))

    def is_registered_connection(self, from_node_type: str, to_node_type: str, edge_type: str) -> bool:
        for from_node_type_in_policy, to_node_type_in_policy in self._edge_type_map[edge_type].connection_data:
            if node_type_matches_type_in_policy(
                from_node_type, from_node_type_in_policy
            ) and node_type_matches_type_in_policy(to_node_type, to_node_type_in_policy):
                return True
        return False


class NodeRegistry:
    def __init__(self, node_type_map: Optional[dict[str, NodeTypeData]] = None):
        self._node_type_map = node_type_map or dict()

    def register_node(
        self,
        node_type: str,
        model: Type[BaseModel],
        instance_limit: Optional[int] = None,
        system_message: Optional[str] = None,
    ):
        if node_type in self._node_type_map:
            raise Exception(f"Node type {node_type} already registered.")
        self._node_type_map[node_type] = NodeTypeData(
            instance_limit=instance_limit, model=model, system_directive=system_message
        )

    def deregister_node(self, node_type: str):
        del self._node_type_map[node_type]

    def disable_node(self, node_type: str):
        self._node_type_map[node_type].write_allowed = False

    def enable_node(self, node_type: str):
        self._node_type_map[node_type].write_allowed = True

    def is_enabled(self, node_type: str) -> bool:
        return self._node_type_map[node_type].write_allowed

    def is_registered(self, node_type: str) -> bool:
        return node_type in self._node_type_map

    def get_node_type_data(self, node_type: str) -> NodeTypeData:
        if node_type not in self._node_type_map:
            raise Exception(f"Node type {node_type} not registered.")
        return self._node_type_map[node_type]

    def clone(self) -> Type["NodeRegistry"]:
        return NodeRegistry(copy.deepcopy(self._node_type_map))
