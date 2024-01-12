"""
Functions that provide boiler plate wrappers for use from Jupyter notebooks,
UI, etc. The logic here might inform updates to the API but it's not clear
where specifically these niceties should live yet, so hedging against API
bloat.
"""

from typing import Iterator, Optional

import gstk.creation.api as creation_api
from gstk.creation.formatters import format_node_for_vectorization
from gstk.creation.graph_registry import CreationNode, GroupProperties
from gstk.creation.group import CreationGroup, new_group
from gstk.graph.interface.graph.graph import Node
from gstk.graph.system_graph_registry import ProjectProperties, SystemEdgeType


class NotFoundError(Exception):
    pass


def print_node_list(node_list: list[Node]):
    for node in node_list:
        print(format_node_for_vectorization(node))
        print()


def get_or_create_project(
    project_id: str, project_name: Optional[str] = None, project_description: Optional[str] = None
):
    if not creation_api.project_id_exists(project_id):
        print(f"Creating project {project_id}")
        project: creation_api.CreationProject = creation_api.new_creation_project(
            project_id, ProjectProperties(id=project_id, description=project_description, name=project_name)
        )
        project.root_group.node.session.commit()
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
    group: CreationGroup
    try:
        group = get_subgroup(parent_group, name)
    except NotFoundError:
        group: CreationGroup = CreationGroup(new_group(parent_group.node, GroupProperties(name=name)))
        group.node.session.commit()
    return group


def get_subgroup(parent_group: CreationGroup, name: str) -> CreationGroup:
    for group in list_subgroups(parent_group):
        if group.node.data.name == name:
            return group
    raise NotFoundError(f"Subgroup {name} not in group {parent_group.node.id}")


def deduplicate_contained_nodes(
    group: CreationGroup, node_type_filter: Optional[list[str]] = None, similarity_threshold: float = 0.95
):
    """
    Slow implementation. In our preliminary use case we can expect to be user input bound
    rather than inner loop bound in the wall clock time, but performance nevertheless is
    to be improved.
    """
    to_delete: set[int] = set()
    scanned: set[int] = set()
    # build index
    # iterate nodes, searching index for each, skipping node if it is to be deleted.
    faiss_index, node_ids = group.node.build_vector_index(
        node_type_filter=node_type_filter, edge_type_filter=[SystemEdgeType.contains]
    )
    vectors = faiss_index.reconstruct_n(0)

    distance_lists, index_lists = faiss_index.search(vectors, len(vectors))
    for i, vector in enumerate(vectors):
        distances: list[float] = distance_lists[i]
        indices: list[int] = index_lists[i]
        scanned.add(node_ids[i])
        for j, distance in enumerate(distances):
            if (
                distance >= similarity_threshold
                and node_ids[indices[j]] not in scanned
                and node_ids[indices[j]] not in to_delete
            ):
                to_delete.add(node_ids[indices[j]])
                print(f"Node {node_ids[i]} is similar to {node_ids[indices[j]]} with distance {distance}")

    print(f"Deleting these nodes: {to_delete}")
    with group.node.graph.create_session() as session:
        for node_id in to_delete:
            group.node.graph.delete_edge_by_nodes(session, group.node.id, node_id)
        session.commit()
    print(f"Deleted {len(to_delete)} nodes")
    group.node.refresh()
