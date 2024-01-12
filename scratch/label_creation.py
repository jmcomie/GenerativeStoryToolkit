from gstk.user_registries.story.graph_registry import StoryEdgeRegistry, StoryNodeRegistry, StoryNodeType
from gstk.graph.registry_context_manager import default_registries
import gstk.creation.api as capi
from gstk.graph.registry_context_manager import current_node_registry
import gstk.graph.registry_context_manager
import asyncio
import gstk.shim

default_registries(StoryNodeRegistry, StoryEdgeRegistry)
creation_label = current_node_registry().get_node_type_data("creation.labels")

project = gstk.shim.get_or_create_project("my project")
root_group = project.root_group

prompt: str = "provide the names of 10 different Manhattan project scientists"


node = asyncio.run(root_group.create("creation.labels", prompt, create_as_orphan=True))



if not node.data.labels:
	node = asyncio.run(root_group.update(node, f"update it to provide the actual labels for the prompt: '{prompt}'"))

