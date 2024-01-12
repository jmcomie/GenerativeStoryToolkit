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

    def walk_tree(
        self,
        descend_into_types: Optional[list[str]] = None,
        yield_node_types: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = None,
    ) -> Iterator[type["Node"]]:
        """
        Cycle-safe but may traverse nodes closer to the project root than the given node.
        That logic will be inherited from a proper graph data object.
        """
        seen: set[int] = {self.id}
        yield from _walk_tree_helper(self, descend_into_types, yield_node_types, edge_type_filter, seen)

    def build_vector_index(
            self,
            node_type_filter: Optional[list[str]] = None,
            edge_type_filter: Optional[list[str]] = None,
            descend_into_types: Optional[list[str]] = None,
    ) -> tuple[faiss.IndexFlatIP, list[int]]:
        """
        Build a vector index for descendent nodes.
        """
        index: Optional[faiss.IndexFlatIP] = None
        node_ids: list[int] = []
        for node in self.walk_tree(
            descend_into_types=descend_into_types,
            yield_node_types=node_type_filter,
            edge_type_filter=edge_type_filter,
        ):
            if node.vector is None:
                continue
            if index is None:
                dim: int = np.array(node.vector).shape[0]
                index = faiss.IndexFlatIP(dim)
            index.add(np.array([node.vector]))
            node_ids.append(node.id)
        assert index is not None
        return index, node_ids

    def find(
        self,
        query_vector: list[float],
        count: int = 10,
        node_type_filter: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = None,
        descend_into_types: Optional[list[str]] = None,
        similarity_threshold: Optional[list[float]] = None,
    ) -> list[tuple[int, float]]:
        index, node_ids = self.build_vector_index(
            node_type_filter=node_type_filter,
            edge_type_filter=edge_type_filter,
            descend_into_types=descend_into_types,
        )
        query_vector = np.array(query_vector)
        # D: distances/similarities, I: indices
        # 0 is the query vector index
        D, I = index.search(np.array([query_vector]), count)            
        assert len(I) == 1
        return [
            (node_ids[I[0][i]], D[0][i])
            for i in range(len(I[0]))
            if similarity_threshold is None or D[i] >= similarity_threshold
        ]

def _walk_tree_helper(
    iter_node: "Node",
    descend_into_types: list[str],
    yield_node_types: list[str],
    edge_type_filter: list[str],
    seen: set,
) -> Iterator[type["Node"]]:
    for edge, node in iter_node.get_out_nodes(edge_type_filter=edge_type_filter):
        assert isinstance(node, Node) and isinstance(edge, Edge)
        if node.id in seen:
            continue
        seen.add(node.id)
        if yield_node_types is None or node_type_matches_type_in_policy_list(node.node_type, tuple(yield_node_types)):
            yield node
        if descend_into_types is None or node_type_matches_type_in_policy_list(
            node.node_type, tuple(descend_into_types)
        ):
            yield from _walk_tree_helper(node, descend_into_types, yield_node_types, edge_type_filter, seen)
