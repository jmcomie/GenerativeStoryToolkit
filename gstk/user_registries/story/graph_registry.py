"""
Graph definitions for story generation.
"""

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from gstk.graph.registry import GraphRegistry, SystemEdgeType


class StoryNodeType(StrEnum):
    character = "story.character"
    religion = "story.religion"
    location = "story.location"
    dialogue = "story.dialogue"
    ALL = "story.*"


@GraphRegistry.node_type(StoryNodeType.character)
class Character(BaseModel):
    """
    Register character attributes.
    """

    _system_message: str = """You are tasked with interpreting directives for the purpose of creating characters.
Each character created can be based a human or non-huamn, rooted in fantasy or human non-fiction tropes,
sci-fi, etc."""
    id: Optional[str] = None
    name: Optional[str] = Field(default=None, description="The name of the character.")
    description: Optional[str] = Field(default=None, description="A description of the character.")
    age: Optional[int] = Field(default=None, description="The age of the character.")
    backstory: Optional[str] = Field(default=None, description="The backstory of the character.")


@GraphRegistry.node_type(StoryNodeType.religion)
class Religion(BaseModel):
    """
    Register religion attributes.
    """

    _system_message: str = (
        "You are tasked with interpreting directives for the purpose of creating religions. "
        "Each religion created can be based on an existing human religion or be a fantasy or science "
        "fiction religion. If a quantity is specified create the number of religions specified. Call "
        "the function more than once.",
    )
    id: Optional[str] = None
    name: Optional[str] = Field(default=None, description="The name of the religion.")
    description: Optional[str] = Field(default=None, description="A description of the religion.")
    backstory: Optional[str] = Field(default=None, description="The backstory of the religion.")


@GraphRegistry.node_type(StoryNodeType.location)
class Location(BaseModel):
    """
    Register location attributes.
    """

    _system_message: str = (
        "You are tasked with interpreting directives for the purpose of creating locations. "
        "Each location created can be based on an existing human location or be a fantasy or science "
        "fiction location."
    )
    id: Optional[str] = None
    name: Optional[str] = Field(default=None, description="The name of the location.")
    description: Optional[str] = Field(default=None, description="A description of the location.")
    backstory: Optional[str] = Field(default=None, description="The backstory of the location.")


class DialogueEntry(BaseModel):
    speaker_node_id: int = Field(default=None, description="The node id corresponding to the speaker")
    text: str = Field(default=None, description="Contiguous lines of dialogue by a single user")


@GraphRegistry.node_type(StoryNodeType.dialogue)
class Dialogue(BaseModel):
    """
    Register dialogue entries. The node id value corresponds to the speaker. Do not inline the speaker names.
    """

    _system_message: str = (
        "You are tasked with writing dialogue between the characters defined or referred to " "in the user prompt."
    )
    entries: list[DialogueEntry] = Field(
        default=None,
        description="A list of dictionaries in which the fields are a speaker_node_id, representing a speaker, "
        "and the text of their line(s).",
    )
