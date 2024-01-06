"""
Formatters for representing graph nodes as strings.
"""

from pydantic import BaseModel

from gstk.graph.interface.graph.graph import Node


def format_instance_for_vectorization(node: Node, prioritized_fields: list[str] = ["name", "description"]) -> str:
    return _format_instance_for_vectorization(node.node_type, node.data, prioritized_fields)


def _format_instance_for_vectorization(
    node_type: str, instance: BaseModel, prioritized_fields: list[str] = ["name", "description"]
) -> str:
    lines: list[str] = [f"Existing {node_type}:", ""]
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


def format_node_for_vectorization(node: Node) -> str:
    lines = ["Node attributes:", ""]
    lines.append(f"Node type: {node.node_type}")
    lines.append(f"Node id: {node.id}")
    lines.extend(["", "", "Node data:", ""])
    lines.append(format_instance_for_vectorization(node))
    return "\n".join(lines)
