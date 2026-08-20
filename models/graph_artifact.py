from dataclasses import dataclass


@dataclass(frozen=True)
class GraphArtifact:
    """The file produced by a graph generation request."""

    image_path: str
    graph_type: str
