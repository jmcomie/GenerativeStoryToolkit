## Graph Types and Registries

The graph type registry contains rules about node and edge types and cardinality. A node or edge type must be registered for it to be used in graph operations.

**Node Type Registration Fields**

| Field  | Type | Description |
|--------|--------------|-----|
| node_type | StrEnum value or string | Indicates the name of the node in the format ```r"(['a-z']+)\.(['a-z']+)"``` in which the first group is the domain and the second is the name.
| instance_limit | int | The number of nodes of this type allowed in a project.
| model | type; Subclass of pydantic BaseModel | The pydantic BaseModel subclass that will be used for OpenAI function schema generation and for serialization, retrieval, type-checking, etc. |
| system_directive | str | The directive to use in guiding the LLM in the creation of the node.
| write_allowed | boolean | Indicates whether the node type can be used by new nodes or only existing ones for reading. Defaults to true.|

**Edge Type Registration Fields**

| Field  | Type | Description |
|--------|--------------|-----|
| edge_type | StrEnum value or string | Indicates the name of the node in the format ```r"(['a-z']+)\.(['a-z']+)"``` in which the first group is the domain and the second is the name.
| edge_cardinality | EdgeCardinality | One of ```ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY``` indicating the outward cardinality.
| write_allowed | boolean | Indicates whether the edge type can be used by new edges or only existing ones for reading. Defaults to true.|


**Edge Type Connections Registration Fields**

| Field  | Type | Description |
|--------|--------------|-----|
| from\_node\_type | StrEnum value or string | Indicates the name of the node in the format ```r"((['a-z']+)\.(['a-z']+|\*))$|(\*)$"``` in which the first group is the domain and the second is the name, or the first and only group is an asterisk. If the value is an asterisk, all node types match. If the name is an asterisk, all names in the domain match.|
| to\_node\_type | StrEnum value or string | Indicates the name of the node in the format ```r"((['a-z']+)\.(['a-z']+|\*))$|(\*)$"``` in which the first group is the domain and the second is the name, or the first and only group is an asterisk. If the value is an asterisk, all node types match. If the name is an asterisk, all names in the domain match. |
| edge\_type | StrEnum value or string | Documented edge type format. |

**Checks**

Edge type node connection checks match rules in this order: no asterisk is a string-wise match against the registered node, asterisk in the name position matches all nodes registered in the domain, a single asterisk matches all nodes.

**Activating node and edge registries**

A node and edge registry is made active via the registry context manager.

Activate the creation registries:

```
from gftk.graph.registry_context_manager import graph_registries
from gftk.creation.graph_registry import SystemNodeRegistry, SystemEdgeRegistry

with current_registries(SystemNodeRegistry, SystemEdgeRegistry):
    # Graph operations.
```

At registration time, if the name is an asterisk all names in the domain match.

### Default Registries

#### System Registry

The system registry defines core project elements.

**Nodes**

| node_type | model | instance\_limit |
| ----- | ------ | ------ |
| system.project | ProjectProperties | 1 |
| system.media | MediaProperties | - |

**Edges**

| edge\_type | edge\_cardinality |
| ----- | ------ |
| system.contains | ONE\_TO\_MANY |
| system.references | MANY\_TO\_MANY |

**Edge Connections**

| edge\_type | from\_node\_type | to\_node\_type |
| ----- | ----- | ----- |
| system.references | * | system.media |
| system.clone | * | * |


### Creation Registry

The creation registry is a superset of the system registry and is for grouping LLM outputs and for creating new creative outputs of any kind.
Context imports are for the purpose of referencing common selected themes anywhere in the system, a supserset of the system registry and has data structures tailored to LLM token context based diffusion.

**Nodes**

| node_type | model | instance\_limit |
| ----- | ------ | ------ |
| creation.message | ProjectProperties | - |
| creation.group | MediaProperties | - |
| creation.group\_collection\_metadata | GroupCollectionMetadata | - |
| creation.group\_summary\_metadata | GroupSummaryMetadata | - |
| creation.chat\_completion | ChatCompletionArguments | - |

**Edge Connections**

| edge\_type | from\_node\_type | to\_node\_type |
| ----- | ----- | ----- |
| system.contains | system.project | creation.group |
| system.contains | creation.group | user.* |
| system.contains | creation.group | creation.group |
| system.references | creation.group | user.*
| system.contains | creation.chat_completion | creation.message |
| system.references | creation.chat_completion | creation.message |
| system.references | creation.message | user.* |
