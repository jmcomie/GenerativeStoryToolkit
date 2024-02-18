"""
API for creation projects.
"""

from functools import cache
from pathlib import Path
from typing import Iterator, Type

from gstk.creation.graph_registry import CreationNode, GroupProperties
from gstk.creation.group import CreationGroup, new_group
from gstk.graph.graph import Graph, Node
from gstk.graph.registry import ProjectProperties, SystemEdgeType, SystemNodeType

ROOT_GROUP_NAME: str = "root"
