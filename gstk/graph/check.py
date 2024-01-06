from typing import Any, Optional

from sqlalchemy.orm.session import Session

from gstk.graph.interface.graph.graph import Graph, Node
from gstk.graph.registry import EdgeCardinality, EdgeTypeData, NodeTypeData
from gstk.graph.registry_context_manager import current_edge_registry, current_node_registry


def check_node_add(
    session: Session, graph: Graph, node_type: str, data: Any, vector: Optional[list[float]] = None
) -> bool:
    node_registry = current_node_registry()
    node_type_data: NodeTypeData = node_registry.get_node_type_data(node_type)

    if not node_type_data.write_allowed:
        raise ValueError(f"Node type {node_type} is not enabled")

    if node_type_data.instance_limit is not None:
        instance_count: int = len(list(graph.list_nodes(session, node_type_filter=[node_type])))
        if instance_count >= node_type_data.instance_limit:
            raise ValueError(f"Node type {node_type} has reached its instance limit")


def check_edge_add(session: Session, graph: Graph, edge_type: str, from_id: int, to_id: int) -> bool:
    edge_type_data: EdgeTypeData = current_edge_registry().get_edge_type_data(edge_type)
    from_node: Node = graph.get_node(session, from_id)
    to_node: Node = graph.get_node(session, to_id)

    if not edge_type_data.write_allowed:
        raise ValueError(f"Edge type {edge_type} is not enabled")

    if edge_type_data.edge_cardinality == EdgeCardinality.ONE_TO_MANY:
        # Look at all in edges of to_node.  Should be no edges of this type.
        for _ in to_node.get_in_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_id} and {to_id}")
    elif edge_type_data.edge_cardinality == EdgeCardinality.ONE_TO_ONE:
        # Look at relevant edges from both nodes. Should be no edges of this type.
        for _ in from_node.get_out_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_id} and {to_id}")
        for _ in to_node.get_in_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_id} and {to_id}")
    elif edge_type_data.edge_cardinality == EdgeCardinality.MANY_TO_MANY:
        pass  # No checks needed.
