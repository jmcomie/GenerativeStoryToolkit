# Generative Story Toolkit

Welcome! This is a pre-alpha passion project released under the MIT license.

### Project Description

The Generative Story Toolkit (GSTK) is an LLM powered library for creating written story components. It uses a novel directed graph structure to provide appropriate context for a chat completion model.  Broadly speaking it facilitates the creation of new story elements, the updating of story elements, incorporating narrative context, and sub-selecting from existing elements as narrative instruments.

### Use Cases

- Development of video game stories, including NPC dialogue and generating assets for game engines.
- World building in fiction writing.
- Summarizing, reconstituting, and storing large texts. By extracting characters, locations, and attributes, texts of any size can be scaled and molded meaningfully by developers/users.
- Brainstorming relationships, affinities, and interests between existing characters or other entities.

### Installation

**Dependencies**

This project requires the following software dependencies: Git, Python (version 3.11 or higher), and [Python Poetry](https://python-poetry.org/docs/#installation).

**Installation Steps**

```bash
git clone https://github.com/jmcomie/GenerativeStoryToolkit
cd GenerativeStoryToolkit
poetry lock && poetry install
```

#### Examples

The `examples` directory contains Jupyter notebooks to help you get started.
