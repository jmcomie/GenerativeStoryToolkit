from collections import deque
import datetime
from functools import cache, reduce
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar, Union

import faiss
import numpy as np
from pydantic import BaseModel
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from gstk.graph.project_locator import ProjectLocator
from gstk.graph.registry import (
    EdgeCardinality,
    EdgeTypeData,
    GraphRegistry,
    NodeTypeData,
    ProjectProperties,
    SystemEdgeType,
    SystemNodeType,
    node_type_matches_type_in_policy_list,
)

# Define the base model
Base = declarative_base()
T = TypeVar('T')



def breadth_first_traversal(root_node: T, node_children_getter: Callable[[T], list[T]]) -> Iterator[T]:
    queue: deque[T] = deque([root_node])
    while queue:
        current_node: T = queue.popleft()
        yield current_node
        queue.extend(node_children_getter(current_node))


class EdgeModel(Base):
    __tablename__ = "edge"
    id = Column(Integer, primary_key=True, autoincrement=True)
    edge_type = Column(String(32))
    from_id = Column(Integer, ForeignKey("node.id", ondelete="CASCADE"))
    to_id = Column(Integer, ForeignKey("node.id", ondelete="CASCADE"))
    sort_idx = Column(Integer, nullable=True)
    in_node = relationship("NodeModel", foreign_keys=[from_id], back_populates="out_edges")
    out_node = relationship("NodeModel", foreign_keys=[to_id], back_populates="in_edges")

    __table_args__ = (
        UniqueConstraint("from_id", "to_id", name="uq_from_id_to_id"),
        Index("idx_from_id", "from_id"),
        Index("idx_to_id", "to_id"),
    )


class NodeModel(Base):
    __tablename__ = "node"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(DateTime, onupdate=datetime.datetime.utcnow)
    deleted_at = Column(DateTime)
    node_type = Column(String(32))
    vector = Column(JSON)
    data = Column(JSON)
    in_edges = relationship("EdgeModel", foreign_keys=[EdgeModel.to_id], back_populates="in_node")
    out_edges = relationship("EdgeModel", foreign_keys=[EdgeModel.from_id], back_populates="out_node")


@cache
def get_engine(path: str):
    engine = create_engine(f"sqlite:///{path}")
    # Create all tables in the engine
    Base.metadata.create_all(engine)
    return engine


def get_session(path: str):
    Session = sessionmaker(bind=get_engine(path))
    return Session()


class Graph:
    db_filename: str = "graph.sqlite"

    def __init__(self, project_resource_location: Path):
        self._project_resource_location = Path(project_resource_location)
        self._project_node_id: Optional[int] = None

    def get_project_node_id(self, session: Session) -> Optional[int]:
        if self._project_node_id is None:
            self._project_node_id = self.get_project_node(session).id
        return self._project_node_id

    def get_project_node(self, session: Session) -> Optional["Node"]:
        for node in session.query(NodeModel).filter_by(node_type=SystemNodeType.PROJECT):
            return Node(node, self, session)

    def add_node(self, session: Session, data: BaseModel, vector: Optional[list[float]] = None) -> "Node":
        node_type: str = list(GraphRegistry.get_node_types(type(data)))[0]
        check_node_add(session, self, node_type)
        node: NodeModel = NodeModel(
            node_type=node_type,
            vector=vector,
            data=data.model_dump(),
        )
        session.add(node)
        session.flush()
        return Node(node, self, session)

    def add_edge(
        self,
        session: Session,
        edge_type: str,
        from_id: int,
        to_id: int,
        sort_idx: Optional[int] = None,
    ) -> EdgeModel:
        check_edge_add(session, self, edge_type, self.get_node(session, from_id), self.get_node(session, to_id))
        edge: EdgeModel = EdgeModel(edge_type=edge_type, from_id=from_id, to_id=to_id, sort_idx=sort_idx)
        session.add(edge)
        session.flush()
        return edge

    def delete_node(self, session: Session, node_id: int):
        node: Optional[NodeModel] = session.query(NodeModel).filter_by(id=node_id).first()
        if node is None:
            return
        if node.out_edges:
            for edge in node.out_edges:
                session.delete(edge)
        session.delete(node)

    def delete_edge(self, session: Session, edge_id: int):
        session.query(EdgeModel).filter_by(id=edge_id).delete()

    def delete_edge_by_nodes(self, session: Session, from_id: int, to_id: int):
        session.query(EdgeModel).filter_by(from_id=from_id, to_id=to_id).delete()

    def get_node(self, session: Session, node_id: int) -> Optional["Node"]:
        result = session.query(NodeModel).filter_by(id=node_id).first()
        return Node(result, self, session) if result else None

    def get_edge_by_id(self, session: Session, edge_id: int) -> Optional[EdgeModel]:
        return session.query(EdgeModel).filter_by(id=edge_id).first()

    def get_edge(self, session: Session, from_id: int, to_id: int, edge_type: str) -> Optional[EdgeModel]:
        result = session.query(EdgeModel).filter_by(from_id=from_id, to_id=to_id, edge_type=edge_type).first()
        return EdgeModel

    def list_nodes(self, session: Session, node_type_filter: Optional[list[str]] = None) -> Iterator["Node"]:
        if node_type_filter is None:
            query = session.query(NodeModel)
        else:
            query = session.query(NodeModel).filter(NodeModel.node_type.in_(node_type_filter))
        for node in query:
            yield Node(node, self, session)

    def list_edges(self, session: Session, edge_type_filter: Optional[list[str]] = None) -> Iterator[EdgeModel]:
        if edge_type_filter is None:
            query = session.query(EdgeModel)
        else:
            query = session.query(EdgeModel).filter(EdgeModel.edge_type.in_(edge_type_filter))
        for edge in query:
            yield edge

    def create_session(self) -> Session:
        self._project_resource_location.mkdir(parents=True, exist_ok=True)
        return get_session(str(self._project_resource_location / self.db_filename))


class Node:
    def __init__(self, sqlalchemy_obj: NodeModel, graph: Graph, session: Session):
        if not isinstance(sqlalchemy_obj, NodeModel):
            raise ValueError(f"sqlalchemy_obj must be a NodeModel, not {type(sqlalchemy_obj)}")
        self._sqlalchemy_obj = sqlalchemy_obj
        self._graph = graph
        self._session = session

    def __repr__(self) -> str:
        return f'<Node type={self.node_type}> object at {hex(id(self))}'

    @property
    def session(self) -> Any:
        return self._session

    @session.setter
    def session(self, value: Session):
        self._session = value
        self._sqlalchemy_obj = self._graph.get_node(self._session, self.id)._sqlalchemy_obj
        if not isinstance(self._sqlalchemy_obj, NodeModel):
            raise ValueError(f"sqlalchemy_obj must be a NodeModel, not {type(self._sqlalchemy_obj)}")

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def id(self) -> int:
        return self._sqlalchemy_obj.id

    @property
    def node_type(self) -> str:
        return self._sqlalchemy_obj.node_type

    @property
    def vector(self) -> Optional[list[float]]:
        return self._sqlalchemy_obj.vector

    @vector.setter
    def vector(self, value: Optional[list[float]]):
        self._sqlalchemy_obj.vector = value

    @property
    def data(self) -> Any:
        # Today: load into registry model.
        return GraphRegistry.get_node_type_data(self.node_type).model(**self._sqlalchemy_obj.data)

    @data.setter
    def data(self, value: BaseModel):
        # Should be an instance of register model -- check.
        if not isinstance(value, GraphRegistry.get_node_type_data(self.node_type).model):
            raise ValueError(f"Node data must be an instance of {self.node_type}, not {type(value)}")
        self._sqlalchemy_obj.data = value.model_dump()

    @property
    def created_at(self) -> datetime.datetime:
        return self._sqlalchemy_obj.created_at

    @property
    def modified_at(self) -> datetime.datetime:
        return self._sqlalchemy_obj.modified_at

    @property
    def deleted_at(self) -> datetime.datetime:
        return self._sqlalchemy_obj.deleted_at

    @deleted_at.setter
    def deleted_at(self, value: datetime.datetime):
        self._sqlalchemy_obj.deleted_at = value

    def _check_node_data_against_filters(self, node: NodeModel, filters: list[dict]) -> bool:
        for filter in filters:
            for key, value in filter.items():
                if key not in node.data or node.data[key] != value:
                    return False
        return True

    def list_children(self, node_types: Optional[list[str]] = None, filters=[]) -> Iterator["Node"]:
        for node in self.walk_tree(yield_node_types=node_types, max_depth=1):
            if node.id == self.id:
                continue
            if not self._check_node_data_against_filters(node._sqlalchemy_obj, filters):
                continue
            yield node

    @property
    def parent(self):
        for edge in self._sqlalchemy_obj.in_edges:
            if edge.edge_type == SystemEdgeType.CONTAINS:
                return Node(edge.in_node, self._graph, self.session)
        return None

    def create_child(self, data: BaseModel) -> "Node":
        node: Node = self._graph.add_node(self.session, data)
        self._graph.add_edge(self.session, SystemEdgeType.CONTAINS, self.id, node.id)
        return node

    def create_child_reference(
        self,
        reference_to: Union["Node", int],
        conflict_filter: Optional[dict] = None,
        overwrite_on_conflict: bool = False,
    ):
        reference_node_id: int = reference_to if isinstance(reference_to, int) else reference_to.id
        edge: EdgeModel = self._graph.add_edge(self.session, SystemEdgeType.CONTAINS, self.id, reference_node_id)

    def delete_child(self, child: Union[int, "Node"]):
        child_id: int = child if isinstance(child, int) else child.id
        self._graph.delete_node(self.session, child_id)

    def delete(self):
        self._graph.delete_node(self.session, self.id)

    def _get_descendent_nodes(self, filters: list[dict], seen: set) -> Iterator["Node"]:
        for edge in self._sqlalchemy_obj.out_edges:
            assert isinstance(edge, EdgeModel)
            if edge.out_node.id in seen:
                continue
            seen.add(edge.out_node.id)
            assert isinstance(edge, EdgeModel)
            assert isinstance(edge.out_node, NodeModel)
            if self._check_node_data_against_filters(edge.out_node, filters):
                yield Node(edge.out_node, self._graph, self.session)
            if edge.edge_type == SystemEdgeType.CONTAINS:
                yield from Node(edge.out_node, self._graph, self.session)._get_descendent_nodes(filters, seen)

    def get_descendent_nodes(self, filters: list[dict]) -> Iterator["Node"]:
        if not filters:
            raise ValueError("Filters must be provided.")
        yield from self._get_descendent_nodes(filters, set())

    def refresh(self):
        assert isinstance(self.session, Session)
        self.session.expire_all()
        self.session.refresh(self._sqlalchemy_obj)

    def save(self):
        assert isinstance(self.session, Session)
        self.session.add(self._sqlalchemy_obj)
        self.session.flush()

    def walk_tree(
        self,
        yield_node_types: Optional[list[str]] = None,
        descend_into_types: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = [SystemEdgeType.CONTAINS, SystemEdgeType.REFERENCES],
        max_depth: Optional[int] = None,
    ) -> Iterator[type["Node"]]:
        """
        Cycle-safe but may traverse nodes closer to the project root than the given node.
        That logic will be inherited from a proper graph data object.
        """
        if max_depth is not None and max_depth < 0:
            return
        yield self
        if max_depth == 0:
            return
        if max_depth is not None:
            max_depth -= 1
        seen: set[int] = {self.id}
        yield from _walk_tree_helper(self, descend_into_types, yield_node_types, edge_type_filter, max_depth, seen)

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
    max_depth: Optional[int],
    seen: set,
) -> Iterator[type["Node"]]:
    if max_depth is not None:
        if max_depth < 0:
            return
        max_depth -= 1
    for edge in iter_node._sqlalchemy_obj.out_edges:
        assert isinstance(edge, EdgeModel)
        # Dangling edges have been occurring. Need to review foreign key constraint
        # behavior.
        if edge.out_node is None:
            continue
        assert isinstance(edge.out_node, NodeModel)
        if edge.out_node.id in seen:
            continue
        seen.add(edge.out_node.id)
        if edge.out_node.deleted_at:
            continue
        if yield_node_types is None or node_type_matches_type_in_policy_list(
            edge.out_node.node_type, tuple(yield_node_types)
        ):
            yield Node(edge.out_node, iter_node.graph, iter_node.session)
        if descend_into_types is None or node_type_matches_type_in_policy_list(
            edge.out_node.node_type, tuple(descend_into_types)
        ):
            yield from _walk_tree_helper(
                Node(edge.out_node, iter_node.graph, iter_node.session),
                descend_into_types,
                yield_node_types,
                edge_type_filter,
                max_depth,
                seen,
            )


def check_node_add(
    session: Session, graph: Graph, node_type: str
) -> bool:
    node_type_data: NodeTypeData = GraphRegistry.get_node_type_data(node_type)

    if not node_type_data.write_allowed:
        raise ValueError(f"Node type {node_type} is not enabled")

    if node_type_data.instance_limit is not None:
        instance_count: int = len(list(graph.list_nodes(session, node_type_filter=[node_type])))
        if instance_count >= node_type_data.instance_limit:
            raise ValueError(f"Node type {node_type} has reached its instance limit")


def check_edge_add(session: Session, graph: Graph, edge_type: str, from_node: Node, to_node: Node) -> bool:
    edge_type_data: EdgeTypeData = GraphRegistry.get_edge_type_data(edge_type)

    if not edge_type_data.write_allowed:
        raise ValueError(f"Edge type {edge_type} is not enabled")

    if edge_type_data.edge_cardinality == EdgeCardinality.ONE_TO_MANY:
        # Look at all in edges of to_node.  Should be no edges of this type.
        for _ in to_node.get_in_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_node.id} and {to_node.id}")
    elif edge_type_data.edge_cardinality == EdgeCardinality.ONE_TO_ONE:
        # Look at relevant edges from both nodes. Should be no edges of this type.
        for _ in from_node.get_out_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_node.id} and {to_node.id}")
        for _ in to_node.get_in_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_node.id} and {to_node.id}")
    elif edge_type_data.edge_cardinality == EdgeCardinality.MANY_TO_MANY:
        pass  # No checks needed.


def new_project(
    project_properties: ProjectProperties,
    project_locator: ProjectLocator,
) -> Node:
    """
    Create a new project with the given project_id and project_properties.
    """
    if project_locator.project_id_exists(project_properties.id):
        raise ValueError(f"Project '{project_properties.id}' already exists")

    graph = Graph(project_locator.get_project_resource_location(project_properties.id))
    project_node: Node
    session = graph.create_session()
    project_node = graph.add_node(session, project_properties)
    session.commit()
    return project_node


def get_project(project_id, project_locator: ProjectLocator) -> Optional[Node]:
    """
    Gets the project with the given project_id. If the project_id does not exist,
    raises a ValueError.
    """
    if not project_locator.project_id_exists(project_id):
        return None
    graph: Graph = Graph(project_locator.get_project_resource_location(project_id))
    session = graph.create_session()
    return graph.get_project_node(session)
