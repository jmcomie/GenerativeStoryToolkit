"""
Graph definitions for story generation.
"""

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from gstk.creation.graph_registry import CreationEdgeRegistry, CreationNode, CreationNodeRegistry
from gstk.graph.system_graph_registry import SystemEdgeType

StoryNodeRegistry = CreationNodeRegistry.clone()
StoryEdgeRegistry = CreationEdgeRegistry.clone()


class StoryNodeType(StrEnum):
    character = "story.character"
    religion = "story.religion"
    location = "story.location"
    dialogue = "story.dialogue"
    ALL = "story.*"


class Character(BaseModel):
    """
    Register character attributes.
    """

    id: Optional[str] = None
    name: Optional[str] = Field(default=None, description="The name of the character.")
    description: Optional[str] = Field(default=None, description="A description of the character.")
    age: Optional[int] = Field(default=None, description="The age of the character.")
    backstory: Optional[str] = Field(default=None, description="The backstory of the character.")


class Religion(BaseModel):
    """
    Register religion attributes.
    """

    id: Optional[str] = None
    name: Optional[str] = Field(default=None, description="The name of the religion.")
    description: Optional[str] = Field(default=None, description="A description of the religion.")
    backstory: Optional[str] = Field(default=None, description="The backstory of the religion.")


class Location(BaseModel):
    """
    Register location attributes.
    """

    id: Optional[str] = None
    name: Optional[str] = Field(default=None, description="The name of the location.")
    description: Optional[str] = Field(default=None, description="A description of the location.")
    backstory: Optional[str] = Field(default=None, description="The backstory of the location.")


class DialogueEntry(BaseModel):
    speaker_node_id: int = Field(default=None, description="The node id corresponding to the speaker")
    text: str = Field(default=None, description="Contiguous lines of dialogue by a single user")


class Dialogue(BaseModel):
    """
    Register dialogue entries. The node id value corresponds to the speaker. Do not inline the speaker names.
    """

    entries: list[DialogueEntry] = Field(
        default=None,
        description="A list of dictionaries in which the fields are a speaker_node_id, representing a speaker, "
        "and the text of their line(s).",
    )


StoryNodeRegistry.register_node(
    StoryNodeType.character,
    model=Character,
    system_message="""You are tasked with interpreting directives for the purpose of creating characters.
Each character created can be based a human or non-huamn, rooted in fantasy or human non-fiction tropes,
sci-fi, etc.""",
)

StoryNodeRegistry.register_node(
    StoryNodeType.religion,
    model=Religion,
    system_message="You are tasked with interpreting directives for the purpose of creating religions. "
    "Each religion created can be based on an existing human religion or be a fantasy or science "
    "fiction religion. If a quantity is specified create the number of religions specified. Call "
    "the function more than once.",
)

StoryNodeRegistry.register_node(
    StoryNodeType.location,
    model=Location,
    system_message="You are tasked with interpreting directives for the purpose of creating locations. "
    "Each location created can be based on an existing human location or be a fantasy or science "
    "fiction location.",
)

StoryNodeRegistry.register_node(
    StoryNodeType.dialogue,
    model=Dialogue,
    system_message="You are tasked with writing dialogue between the characters defined or referred to "
    "in the user prompt.",
)

StoryEdgeRegistry.register_connection_types(
    CreationNode.group,
    StoryNodeType.ALL,
    [SystemEdgeType.contains, SystemEdgeType.references],
)
