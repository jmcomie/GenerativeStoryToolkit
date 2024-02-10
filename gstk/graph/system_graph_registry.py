"""
Establishes the core system graph structures. The node and edge registries can be
cloned and modified to create custom graph structures.
"""

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from gstk.graph.registry import ALL_NODES, EdgeCardinality, EdgeRegistry, NodeRegistry

SystemNodeRegistry = NodeRegistry()
SystemEdgeRegistry = EdgeRegistry()
