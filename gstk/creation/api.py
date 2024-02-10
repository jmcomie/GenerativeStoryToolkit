"""
API for creation projects.
"""

from functools import cache
from pathlib import Path
from typing import Iterator, Type

from gstk.creation.graph_registry import CreationEdgeRegistry, CreationNode, CreationNodeRegistry, GroupProperties
from gstk.creation.group import CreationGroup, new_group
from gstk.graph.graph import SQLiteGraph
from gstk.graph.interface.graph.graph import Graph, Node
from gstk.graph.interface.resource_locator.resource_locator import ResourceLocator
from gstk.graph.local_file import LocalFileLocator
from gstk.graph.registry_context_manager import graph_registries
from gstk.graph.system_graph_registry import ProjectProperties, SystemEdgeType, SystemNodeType

ROOT_GROUP_NAME: str = "root"


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

    def __init__(self, project_node: Node, resource_location: Path):
        if project_node.node_type != SystemNodeType.project:
            raise ValueError(f"project_node must be a project node, not {project_node.node_type}")
        self._project_node = project_node
        self._resource_location = resource_location

    @property
    @creation_registry
    def root_group(self) -> CreationGroup:
        for _edge, node in self._project_node.get_out_nodes(
            edge_type_filter=[SystemEdgeType.contains], node_type_filter=[CreationNode.group]
        ):
            if node.data.name == ROOT_GROUP_NAME:
                return CreationGroup(node)
        raise ValueError("Project has no root group")

    @property
    def project_node(self) -> Node:
        return self._project_node

    @property
    def graph(self) -> Graph:
        return self._project_node.graph

    def get_node(self, node_id: int) -> Node:
        return self.graph.get_node(self._project_node.session, node_id)

    @property
    def properties(self):
        return self._project_node.data

    @property
    def supplemental_data(self):
        assert isinstance(self._project_node.data, ProjectProperties)
        return self._project_node.data.supplemental_data

    @property
    def resource_location(self) -> ResourceLocator:
        return self._resource_location


@cache
def _get_resource_locator() -> ResourceLocator:
    """
    Default resource locator for creation is LocalFileLocator.
    """
    return LocalFileLocator()


def _get_graph(project_id: str, resource_locator: ResourceLocator = _get_resource_locator()):
    """
    This returns the graph for the given project_id. The graph is the data storage
    and traversal substrate for gstk projects.
    """
    # Load the graph and then load the project
    return SQLiteGraph(resource_locator.get_project_resource_location(project_id))


@creation_registry
def get_creation_project(
    project_id, resource_locator: ResourceLocator = _get_resource_locator(), project_class: Type = CreationProject
) -> CreationProject:
    """
    Gets the project with the given project_id. If the project_id does not exist,
    raises a ValueError.
    """
    if not resource_locator.project_id_exists(project_id):
        raise ValueError(f"Project '{project_id}' not found")
    graph = _get_graph(project_id, resource_locator=resource_locator)
    session = graph.create_session()
    return project_class(
        graph.get_project_node(session), resource_location=resource_locator.get_project_resource_location(project_id)
    )


@creation_registry
def list_creation_project_ids(resource_locator: ResourceLocator = _get_resource_locator()) -> Iterator[str]:
    """
    Lists the project_ids.
    """
    yield from resource_locator.list_project_ids()


def project_id_exists(project_id: str, resource_locator: ResourceLocator = _get_resource_locator()) -> bool:
    """
    Returns boolean indicating whether the project_id exists.
    """
    return resource_locator.project_id_exists(project_id)


@creation_registry
def new_creation_project(
    project_id: str,
    project_properties: ProjectProperties,
    resource_locator: ResourceLocator = _get_resource_locator(),
    project_class: Type = CreationProject,
) -> CreationProject:
    """
    Create a new project with the given project_id and project_properties.
    """
    if resource_locator.project_id_exists(project_id):
        raise ValueError(f"Project '{project_id}' already exists")

    graph = _get_graph(project_id, resource_locator=resource_locator)
    project_node: Node
    session = graph.create_session()
    project_node = graph.add_node(session, SystemNodeType.project, project_properties)
    new_group(project_node, GroupProperties(name=ROOT_GROUP_NAME))
    session.commit()
    return project_class(project_node, resource_location=resource_locator.get_project_resource_location(project_id))


def delete_creation_project(project_id: str, resource_locator: ResourceLocator = _get_resource_locator()):
    """
    Delete the project with the given project_id.
    """
    if not resource_locator.project_id_exists(project_id):
        raise ValueError(f"Project '{project_id}' not found")
    resource_locator.delete_project(project_id)
