import os
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from generators.api import SUPPORTED_GRAPH_TYPES, generate_graph
from models.graph_artifact import GraphArtifact
from models.graph_settings import GraphSettings


class GraphApiTests(unittest.TestCase):
    def setUp(self):
        self.created_paths: list[Path] = []

    def tearDown(self):
        for path in self.created_paths:
            path.unlink(missing_ok=True)

    def generate(self, graph_type, *, equation=None, equations=None):
        filename = f"api-{graph_type.lower()}.png"
        settings = GraphSettings(output_name=filename, image_dpi=30)
        artifact = generate_graph(
            graph_type=graph_type,
            equation=equation,
            equations=equations,
            settings=settings,
        )
        self.created_paths.append(Path(artifact.image_path))
        return artifact

    def test_supported_graph_types_are_centralised(self):
        self.assertEqual(
            SUPPORTED_GRAPH_TYPES,
            (
                "Linear",
                "Quadratic",
                "Exponential",
                "Hyperbola",
                "Cubic",
                "Logarithmic",
                "Sine",
                "Cosine",
                "Tangent",
                "Circle",
                "Mixed",
            ),
        )

    def test_individual_graphs_return_non_empty_artifacts(self):
        cases = {
            "Linear": "2*x + 1",
            "Quadratic": "x**2 - 4*x + 3",
            "Hyperbola": "2/(x - 1) + 3",
            "Sine": "sin(x)",
            "Circle": "x**2 + y**2 = 25",
        }
        for graph_type, equation in cases.items():
            with self.subTest(graph_type=graph_type):
                artifact = self.generate(graph_type, equation=equation)
                self.assertIsInstance(artifact, GraphArtifact)
                self.assertEqual(artifact.graph_type, graph_type)
                self.assertTrue(os.path.isfile(artifact.image_path))
                self.assertGreater(os.path.getsize(artifact.image_path), 0)

    def test_mixed_graph_returns_non_empty_artifact(self):
        artifact = self.generate(
            "Mixed",
            equations=["2*x + 1", "x**2 - 4"],
        )
        self.assertEqual(artifact.graph_type, "Mixed")
        self.assertTrue(os.path.isfile(artifact.image_path))
        self.assertGreater(os.path.getsize(artifact.image_path), 0)

    def test_missing_individual_equation_fails(self):
        with self.assertRaisesRegex(ValueError, "An equation is required"):
            generate_graph("Linear", GraphSettings())

    def test_mixed_requires_exactly_two_equations(self):
        for equations in (None, [], ["x"], ["x", "x**2", "2**x"]):
            with self.subTest(equations=equations), self.assertRaisesRegex(
                ValueError, "exactly two equations"
            ):
                generate_graph("Mixed", GraphSettings(), equations=equations)

    def test_unsupported_graph_type_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported graph type: Piecewise"):
            generate_graph("Piecewise", GraphSettings(), equation="x")


if __name__ == "__main__":
    unittest.main()
