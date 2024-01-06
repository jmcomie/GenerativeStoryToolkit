Groups

GroupProperties


Project can only contain groups.

## Group

can the contains/references distinction combined with out node types suffice for all distinctions needed?

would it make sense to allow the user to indicate that a user.* instance should be selectable / findable / etc, regardless of contains/references in edge. Could be stored in a extended_attrs field of the base group data, with "node", "edge" section,

\# remove is_exported field

Can contain:
creation.message
user.group


**async def create\_entity(node\_type, prompt):**

Sends message to chat completion with the model and system message for the node_type followed by all out nodes applicable to chat completion. Created data is added as the output node of a system.contains edge type. If LLM returns no function call response an entry is added to the project error log containing the prompt and the llm response or error, and no entry is added to group.

Out nodes are retrieved for chat completion only if the edge to them is system.contains.

**async def select(prompt: str, node\_type\_filter: Optional[list[str]] = None)**

#

Has LLM select among the nodes in the group connected via a system.contains edge. If node\_type\_filter is provided only those node types will be included in the selection. Reason is added to the debug log.



**find**



Get messages for chat completion.

```

```

**def get\_messages\_for\_chat\_completion**
