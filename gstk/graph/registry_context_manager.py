import threading
from contextlib import contextmanager
from typing import Optional

from gstk.graph.registry import EdgeRegistry, NodeRegistry

_ctx = threading.local()
_ctx.registries_list: list[tuple[NodeRegistry, EdgeRegistry]] = []
_ctx.default_registries: tuple[NodeRegistry, EdgeRegistry] = None


@contextmanager
def graph_registries(node_registry: NodeRegistry, edge_registry: EdgeRegistry) -> None:
    _ctx.registries_list.append((node_registry, edge_registry))
    yield
    _ctx.registries_list.pop()


def default_registries(node_registry: NodeRegistry, edge_registry: EdgeRegistry):
    _ctx.default_registries = (node_registry, edge_registry)


def _current_registry(index: int) -> Optional[NodeRegistry | EdgeRegistry]:
    if not _ctx.registries_list and not _ctx.default_registries:
        raise RuntimeError("No active registry context.")
    if _ctx.registries_list:
        return _ctx.registries_list[-1][index]
    if _ctx.default_registries:
        return _ctx.default_registries[index]


def current_node_registry() -> Optional[NodeRegistry]:
    return _current_registry(0)


def current_edge_registry() -> Optional[EdgeRegistry]:
    return _current_registry(1)
