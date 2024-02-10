from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import faiss
import numpy as np

from gstk.graph.registry import node_type_matches_type_in_policy_list


class Graph(ABC):
    @abstractmethod
    def get_project_node(self, session: Any) -> Optional["Node"]:
        pass

    @abstractmethod
    def add_node(self, session, node_type: str, data: Any, vector: Optional[list[float]] = None) -> "Node":
        pass

    @abstractmethod
    def add_edge(
        self,
        session: Any,
        edge_type: str,
        from_id: int,
        to_id: int,
        sort_idx: Optional[int] = None,
    ) -> "Edge":
        pass

    @abstractmethod
    def get_node(self, session: Any, node_id: int) -> "Node":
        pass

    @abstractmethod
    def list_nodes(self, session: Any, node_type_filter: Optional[list[str]] = None) -> Iterator["Node"]:
        pass

    @abstractmethod
    def list_edges(self, session: Any, edge_type_filter: Optional[list[str]] = None) -> Iterator["Edge"]:
        pass

    @abstractmethod
    def get_edge(self, session: Any, edge_id: int) -> "Edge":
        pass

    @abstractmethod
    def create_session(self) -> Any:
        pass


class Edge(ABC):
    @property
    @abstractmethod
    def graph(self) -> Graph:
        pass

    @property
    @abstractmethod
    def id(self) -> int:
        pass

    @property
    @abstractmethod
    def edge_type(self) -> str:
        pass

    @property
    @abstractmethod
    def from_id(self) -> int:
        pass

    @property
    @abstractmethod
    def to_id(self) -> int:
        pass

    @property
    @abstractmethod
    def sort_idx(self) -> int:
        pass

    @abstractmethod
    def refresh(self, session: Any):
        pass

    @abstractmethod
    def save(self, session: Any):
        pass


class Node(ABC):
    @property
    @abstractmethod
    def graph(self) -> Graph:
        pass

    @property
    @abstractmethod
    def id(self) -> int:
        pass

    @property
    @abstractmethod
    def node_type(self) -> str:
        pass

    @property
    @abstractmethod
    def data(self) -> dict:
        pass

    @property
    @abstractmethod
    def vector(self) -> Optional[list[float]]:
        pass

    @property
    @abstractmethod
    def created_at(self) -> int:
        pass

    @property
    @abstractmethod
    def modified_at(self) -> Optional[int]:
        pass

    @property
    @abstractmethod
    def deleted_at(self) -> Optional[int]:
        pass

    @abstractmethod
    def get_in_nodes(self, session: Any, edge_type_filter: list[str]) -> list[tuple["Edge", "Node"]]:
        pass

    @abstractmethod
    def get_out_nodes(self, session: Any, edge_type_filter: list[str]) -> list[tuple["Edge", "Node"]]:
        pass

    @abstractmethod
    def add_out_edge(self, session: Any, edge_type: str, node_id: int) -> "Edge":
        pass

    @abstractmethod
    def add_in_edge(self, session: Any, edge_type: str, node_id: int) -> "Edge":
        pass

    @abstractmethod
    def add_out_node(
        self, session: Any, node_type: str, data: Any, edge_type: str, vector: Optional[list[float]] = None
    ) -> tuple["Edge", "Node"]:
        pass

    @abstractmethod
    def add_in_node(
        self, session: Any, node_type: str, data: Any, edge_type: str, vector: Optional[list[float]] = None
    ) -> tuple["Edge", "Node"]:
        pass

    @abstractmethod
    def refresh(self, session: Any):
        pass

    @abstractmethod
    def save(self, session: Any):
        pass

