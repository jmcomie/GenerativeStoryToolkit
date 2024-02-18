"""
Formatters for representing graph nodes as strings.
"""

from pydantic import BaseModel

from gstk.graph.graph import Node


def format_instance_for_vectorization(
    instance: BaseModel, prioritized_fields: list[str] = ["name", "description"]
) -> str:
    lines: list[str] = []
    prioritized_fields: list[str] = ["name", "description"]
    for field_name in prioritized_fields:
        if field_name in instance.model_fields:
            if not getattr(instance, field_name):
                continue
            lines.append(f"{field_name}: {getattr(instance, field_name)}")

    for field_name in instance.model_fields:
        if field_name not in prioritized_fields:
            if not getattr(instance, field_name):
                continue
            lines.append(f"{field_name}: {getattr(instance, field_name)}")

    return "\n".join(lines)


def format_node_for_creation_context(node: Node):
    """
    Formatter for creation/updating. For instance, to shape the tone or hedge against
    duplication.
    """
    return f"Existing {node.node_type}:\n" + format_instance_for_vectorization(node.data)


def format_node_for_selection_context(node: Node):
    """
    Formatter for selecting among a set of nodes.
    """
    lines = ["Node attributes:", ""]
    lines.append(f"Node type: {node.node_type}")
    lines.append(f"Node id: {node.id}")
    lines.extend(["", "", "Node data:", ""])
    lines.append(format_instance_for_vectorization(node))
    return "\n".join(lines)


def format_node_for_vectorization(node: Node, prioritized_fields: list[str] = ["name", "description"]):
    """
    Formatter for vectorization.
    """
    return format_instance_for_vectorization(node.data, prioritized_fields=prioritized_fields)
