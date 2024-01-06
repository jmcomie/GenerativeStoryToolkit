"""
Functions that provide boiler plate wrappers for use from Jupyter notebooks,
UI, etc. The logic here might inform updates to the API but it's not clear
where specifically these niceties should live yet, so hedging against API
bloat.
"""

from typing import Iterator, Optional

import gstk.creation.api as creation_api
from gstk.creation.formatters import format_node_for_vectorization
from gstk.creation.graph_registry import CreationNode, GroupCollectionData, GroupProperties
from gstk.creation.group import CreationGroup, new_group
from gstk.graph.interface.graph.graph import Node
from gstk.graph.registry_context_manager import default_registries
from gstk.graph.system_graph_registry import ProjectProperties
from gstk.user_registries.story.graph_registry import StoryEdgeRegistry, StoryNodeRegistry


def print_node_list(node_list: list[Node]):
    for node in node_list:
        print(format_node_for_vectorization(node))


def get_or_create_project(
    project_id: str, project_name: Optional[str] = None, project_description: Optional[str] = None
):
    default_registries(StoryNodeRegistry, StoryEdgeRegistry)
    if not creation_api.project_id_exists(project_id):
        return creation_api.new_creation_project(
            project_id, ProjectProperties(id=project_id, description=project_description, name=project_name)
        )
    return creation_api.get_creation_project(project_id)


def print_group_nodes(
    group: CreationGroup, node_type_filter: Optional[list[str]] = None, edge_type_filter: Optional[list[str]] = None
):
    iterator = group.node.get_out_nodes(node_type_filter=node_type_filter, edge_type_filter=edge_type_filter)
    print_node_list(list(zip(*iterator))[1])


def get_out_node_count(
    group: CreationGroup, node_type_filter: Optional[list[str]] = None, edge_type_filter: Optional[list[str]] = None
):
    return len(list(group.node.get_out_nodes(node_type_filter=node_type_filter, edge_type_filter=edge_type_filter)))


def list_subgroups(group: CreationGroup) -> Iterator[CreationGroup]:
    for edge, node in group.node.get_out_nodes(node_type_filter=[CreationNode.group]):
        yield CreationGroup(node)


def get_or_create_subgroup(parent_group: CreationGroup, name: str) -> CreationGroup:
    for group in list_subgroups(parent_group):
        if group.node.data.name == name:
            return group
    group: CreationGroup = CreationGroup(
        new_group(parent_group.node, GroupProperties(name=name), GroupCollectionData())
    )
    group.node.session.commit()
    return group
