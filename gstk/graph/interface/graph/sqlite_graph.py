import datetime
from functools import cache
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from gstk.graph.check import check_edge_add, check_node_add
from gstk.graph.interface.graph.graph import Edge, Graph, Node
from gstk.graph.registry import node_type_matches_type_in_policy_list
from gstk.graph.registry_context_manager import current_node_registry
from gstk.graph.system_graph_registry import SystemNodeType

# Define the base model
Base = declarative_base()


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


class SQLiteGraph(Graph):
    db_filename: str = "graph.sqlite"
    media_directory: str = "media"

    def __init__(self, project_resource_location: Path):
        self._project_resource_location = Path(project_resource_location)

    def get_project_node(self, session: Session) -> Optional["SQLiteNode"]:
        for node in session.query(NodeModel).filter_by(node_type=SystemNodeType.project):
            return SQLiteNode(node, self, session)

    def add_node(
        self, session: Session, node_type: str, data: BaseModel, vector: Optional[list[float]] = None
    ) -> "SQLiteNode":
        check_node_add(session, self, node_type, data, vector=vector)
        node: NodeModel = NodeModel(
            node_type=node_type,
            vector=vector,
            data=data.model_dump(),
        )
        session.add(node)
        session.flush()
        return SQLiteNode(node, self, session)

    def add_edge(
        self,
        session: Session,
        edge_type: str,
        from_id: int,
        to_id: int,
        sort_idx: Optional[int] = None,
    ) -> "SQLiteEdge":
        check_edge_add(session, self, edge_type, from_id, to_id)
        edge: EdgeModel = EdgeModel(edge_type=edge_type, from_id=from_id, to_id=to_id, sort_idx=sort_idx)
        session.add(edge)
        session.flush()
        return SQLiteEdge(edge, self, session)

    def delete_node(self, session: Session, node_id: int):
        node: Optional[NodeModel] = session.query(NodeModel).filter_by(id=node_id).first()
        if node is None:
            return
        if node.out_edges:
            raise ValueError("Cannot delete node with outgoing edges")
        session.delete(node)

    def delete_edge(self, session: Session, edge_id: int):
        session.query(EdgeModel).filter_by(id=edge_id).delete()

    def delete_edge_by_nodes(self, session: Session, from_id: int, to_id: int):
        session.query(EdgeModel).filter_by(from_id=from_id, to_id=to_id).delete()

    def get_node(self, session: Session, node_id: int) -> Optional["SQLiteNode"]:
        result = session.query(NodeModel).filter_by(id=node_id).first()
        return SQLiteNode(result, self, session) if result else None

    def get_edge_by_id(self, session: Session, edge_id: int) -> Optional["SQLiteEdge"]:
        result = session.query(EdgeModel).filter_by(id=edge_id).first()
        return SQLiteEdge(result, self, session) if result else None

    def get_edge(self, session: Session, from_id: int, to_id: int, edge_type: str) -> Optional["SQLiteEdge"]:
        result = session.query(EdgeModel).filter_by(from_id=from_id, to_id=to_id, edge_type=edge_type).first()
        return SQLiteEdge(result, self, session) if result else None

    def list_nodes(self, session: Session, node_type_filter: Optional[list[str]] = None) -> Iterator["SQLiteNode"]:
        if node_type_filter is None:
            query = session.query(NodeModel)
        else:
            query = session.query(NodeModel).filter(NodeModel.node_type.in_(node_type_filter))
        for node in query:
            yield SQLiteNode(node, self, session)

    def list_edges(self, session: Session, edge_type_filter: Optional[list[str]] = None) -> Iterator["SQLiteEdge"]:
        if edge_type_filter is None:
            query = session.query(EdgeModel)
        else:
            query = session.query(EdgeModel).filter(EdgeModel.edge_type.in_(edge_type_filter))
        for edge in query:
            yield SQLiteEdge(edge, self, session)

    def create_session(self) -> Session:
        self._project_resource_location.mkdir(parents=True, exist_ok=True)
        return get_session(str(self._project_resource_location / self.db_filename))


class SQLiteNode(Node):
    def __init__(self, sqlalchemy_obj: NodeModel, graph: SQLiteGraph, session: Session):
        if not isinstance(sqlalchemy_obj, NodeModel):
            raise ValueError(f"sqlalchemy_obj must be a NodeModel, not {type(sqlalchemy_obj)}")
        self._sqlalchemy_obj = sqlalchemy_obj
        self._graph = graph
        self._session = session

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
    def graph(self) -> SQLiteGraph:
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
        return current_node_registry().get_node_type_data(self.node_type).model(**self._sqlalchemy_obj.data)

    @data.setter
    def data(self, value: BaseModel):
        # Should be an instance of register model -- check.
        if not isinstance(value, current_node_registry().get_node_type_data(self.node_type).model):
            raise ValueError(f"Node data must be an instance of {self.node_type}, not {type(value)}")
        print("setting data")
        print(type(value))
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

    def get_out_nodes(
        self, edge_type_filter: Optional[list[str]] = None, node_type_filter: Optional[list[str]] = None
    ) -> Iterator[tuple["SQLiteEdge", "SQLiteNode"]]:
        for edge in self._sqlalchemy_obj.out_edges:
            assert isinstance(edge, EdgeModel)
            assert isinstance(edge.out_node, NodeModel)
            if (edge_type_filter is None or edge.edge_type in edge_type_filter) and (
                node_type_filter is None
                or node_type_matches_type_in_policy_list(edge.out_node.node_type, tuple(node_type_filter))
            ):
                yield SQLiteEdge(edge, self._graph, self.session), SQLiteNode(edge.out_node, self._graph, self.session)

    def get_in_nodes(
        self, edge_type_filter: Optional[list[str]] = None, node_type_filter: Optional[list[str]] = None
    ) -> list[tuple["SQLiteEdge", "SQLiteNode"]]:
        for edge in self._sqlalchemy_obj.in_edges:
            assert isinstance(edge, EdgeModel)
            assert isinstance(edge.in_node, NodeModel)
            if (edge_type_filter is None or edge.edge_type in edge_type_filter) and (
                node_type_filter is None
                or node_type_matches_type_in_policy_list(edge.in_node.node_type, tuple(node_type_filter))
            ):
                yield SQLiteEdge(edge, self._graph, self.session), SQLiteNode(edge.in_node, self._graph, self.session)

    def add_out_edge(self, edge_type: str, to_node: "SQLiteNode", sort_idx: Optional[int] = None) -> "SQLiteEdge":
        self._graph.add_edge(self._session, edge_type, self.id, to_node.id, sort_idx=sort_idx)

    def add_in_edge(self, edge_type: str, from_node: "SQLiteNode", sort_idx: Optional[int] = None) -> "SQLiteEdge":
        return SQLiteEdge(self._graph.add_edge(self.session, edge_type, from_node.id, self.id, sort_idx=sort_idx))

    def add_out_node(
        self, node_type: str, data: Any, edge_type: str, vector: Optional[list[float]] = None
    ) -> tuple["SQLiteEdge", "SQLiteNode"]:
        node: NodeModel = self._graph.add_node(self.session, node_type, data, vector=vector)
        edge: EdgeModel = self._graph.add_edge(self.session, edge_type, self.id, node.id)
        return edge, node

    def add_in_node(
        self, node_type: str, data: Any, edge_type: str, vector: Optional[list[float]] = None
    ) -> tuple["SQLiteEdge", "SQLiteNode"]:
        node: NodeModel = self._graph.add_node(self.session, node_type, data, vector=vector)
        edge: EdgeModel = self._graph.add_edge(self.session, edge_type, node.id, self.id)
        return edge, node

    def refresh(self):
        assert isinstance(self.session, Session)
        self.session.expire_all()
        self.session.refresh(self._sqlalchemy_obj)

    def save(self):
        assert isinstance(self.session, Session)
        self.session.add(self._sqlalchemy_obj)
        self.session.flush()


class SQLiteEdge(Edge):
    def __init__(self, sqlalchemy_obj: EdgeModel, graph: SQLiteGraph, session: Session):
        self._sqlalchemy_obj = sqlalchemy_obj
        self._graph = graph
        self._session = session

    @property
    def session(self) -> Any:
        return self._session

    @session.setter
    def session(self, value: Session):
        self._session = value
        self._sqlalchemy_obj = self._graph.get_node(self._session, self.id)

    @property
    def graph(self) -> SQLiteGraph:
        return self._graph

    @property
    def id(self) -> int:
        return self._sqlalchemy_obj.id

    @property
    def edge_type(self) -> str:
        return self._sqlalchemy_obj.edge_type

    @property
    def from_id(self) -> int:
        return self._sqlalchemy_obj.from_id

    @property
    def to_id(self) -> int:
        return self._sqlalchemy_obj.to_id

    @property
    def sort_idx(self) -> int:
        return self._sqlalchemy_obj.sort_idx

    @sort_idx.setter
    def sort_idx(self, value: int):
        self._sqlalchemy_obj.sort_idx = value

    def refresh(self):
        assert isinstance(self.session, Session)
        self.session.refresh(self._sqlalchemy_obj)

    def save(self):
        assert isinstance(self.session, Session)
        self.session.add(self._sqlalchemy_obj)
        self.session.flush()
