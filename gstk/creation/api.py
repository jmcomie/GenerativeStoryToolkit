"""
API for creation projects.
"""

from functools import cache
from typing import Iterator

from gstk.creation.graph_registry import (
    CreationEdgeRegistry,
    CreationNode,
    CreationNodeRegistry,
    GroupCollectionData,
    GroupProperties,
)
from gstk.creation.group import CreationGroup, new_group
from gstk.graph.interface.graph.graph import Node
from gstk.graph.interface.graph.sqlite_graph import SQLiteGraph
from gstk.graph.interface.resource_locator.local_file import LocalFileLocator
from gstk.graph.interface.resource_locator.resource_locator import ResourceLocator
from gstk.graph.registry_context_manager import graph_registries
from gstk.graph.system_graph_registry import ProjectProperties, SystemEdgeType, SystemNodeType


def creation_registry(fn):
    """
    Decorator to ensure that the CreationNodeRegistry and CreationEdgeRegistry
    are used when calling the function.
    """

    def wrapper(*args, **kwargs):
        with graph_registries(CreationNodeRegistry, CreationEdgeRegistry):
            return fn(*args, **kwargs)

    return wrapper


class CreationProject:
    """
    A Creation Project. This is the top-level creative object within a GFTK project.
    """

    def __init__(self, project_node: Node):
        if project_node.node_type != SystemNodeType.project:
            raise ValueError(f"project_node must be a project node, not {project_node.node_type}")
        self._project_node = project_node

    @property
    @creation_registry
    def root_group(self) -> CreationGroup:
        for _edge, node in self._project_node.get_out_nodes(
            edge_type_filter=[SystemEdgeType.contains], node_type_filter=[CreationNode.group]
        ):
            if node.data.name == "root":
                return CreationGroup(node)
        raise ValueError("Project has no root group")


@cache
def _get_resource_locator() -> ResourceLocator:
    """
    Default resource locator for creation is LocalFileLocator.
    """
    return LocalFileLocator()


def _get_graph(project_id: str):
    """
    This returns the graph for the given project_id. The graph is the data storage
    and traversal substrate for gstk projects.
    """
    # Load the graph and then load the project
    return SQLiteGraph(_get_resource_locator().get_project_resource_location(project_id))


@creation_registry
def get_creation_project(project_id) -> CreationProject:
    """
    Gets the project with the given project_id. If the project_id does not exist,
    raises a ValueError.
    """
    if not _get_resource_locator().project_id_exists(project_id):
        raise ValueError(f"Project `{project_id}` not found")
    graph = _get_graph(project_id)
    session = graph.create_session()
    return CreationProject(graph.get_project_node(session))


@creation_registry
def list_creation_project_ids() -> Iterator[str]:
    """
    Lists the project_ids.
    """
    yield from _get_resource_locator().list_project_ids()


def project_id_exists(project_id: str) -> bool:
    """
    Returns boolean indicating whether the project_id exists.
    """
    return _get_resource_locator().project_id_exists(project_id)


@creation_registry
def new_creation_project(project_id: str, project_properties: ProjectProperties) -> CreationProject:
    """
    Create a new project with the given project_id and project_properties.
    """
    if _get_resource_locator().project_id_exists(project_id):
        raise ValueError(f"Project `{project_id}` already exists")
    graph = _get_graph(project_id)
    project_node: Node
    session = graph.create_session()
    project_node = graph.add_node(session, SystemNodeType.project, project_properties)
    new_group(project_node, GroupProperties(name="root"), GroupCollectionData())
    session.commit()
    return CreationProject(project_node)
