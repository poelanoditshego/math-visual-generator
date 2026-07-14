import unittest
from unittest.mock import patch

import matplotlib
import sympy as sp

matplotlib.use("Agg")

from generators.exponential import horizontal_asymptote
from generators.graph_helpers import graph_label
from generators.mixed import (
    create_mixed_graph,
    detect_graph_type,
    find_real_roots,
    parse_mixed_expression,
)
from models.graph_settings import GraphSettings


class MixedGraphTests(unittest.TestCase):
    def setUp(self):
        self.x = sp.Symbol("x", real=True)

    def expression(self, text):
        return sp.sympify(text, locals={"x": self.x})

    def test_detects_supported_types_and_order(self):
        expected = {
            "x + 1": "Linear",
            "x**2 - 1": "Quadratic",
            "2**x": "Exponential",
            "(1/2)**x + 1": "Exponential",
        }
        for text, graph_type in expected.items():
            with self.subTest(text=text):
                self.assertEqual(detect_graph_type(self.expression(text), self.x), graph_type)
        self.assertEqual(parse_mixed_expression("y = 2**x", 1)[2], "Exponential")
        self.assertEqual(parse_mixed_expression("y = x + 1", 2)[2], "Linear")

    def test_rejects_unsupported_expressions_with_equation_number(self):
        for text in ("sin(x)", "log(x)", "1/x", "tan(x)", "x**3"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "Equation 2"):
                parse_mixed_expression(text, 2)

    def test_finds_all_visible_intersections(self):
        cases = [
            ("x + 1", "2**x", [0.0, 1.0]),
            ("2*x - 3", "2**x - 4", [0.0, 2.65986118]),
            ("-x + 4", "3**(x - 1)", [1.74155181]),
            ("0.5*x - 2", "(1/2)**x + 1", [6.03059428]),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                roots = find_real_roots(
                    self.expression(left) - self.expression(right), self.x, -10, 10
                )
                self.assertEqual(len(roots), len(expected))
                for actual, wanted in zip(roots, expected):
                    self.assertAlmostEqual(actual, wanted, places=6)

    def test_exponential_intercepts_and_asymptotes(self):
        self.assertEqual(find_real_roots(self.expression("2**x"), self.x, -10, 10), [])
        self.assertAlmostEqual(find_real_roots(self.expression("2**x - 4"), self.x, -10, 10)[0], 2.0)
        self.assertEqual(horizontal_asymptote(self.expression("2**x + 3"), self.x), 3.0)

    def test_graph_labels_follow_input_order(self):
        first = self.expression("2**x")
        second = self.expression("x + 1")
        self.assertIn("f(x)", graph_label(first, 0, "Function equation"))
        self.assertIn("g(x)", graph_label(second, 1, "Function equation"))

    @patch("generators.mixed.plt.savefig")
    def test_required_pairs_and_existing_polynomial_pairs_render(self, savefig):
        settings = GraphSettings(
            x_min=-10, x_max=10, y_min=-10, y_max=10,
            show_point_labels=True, additional_x_values=[0, 1],
            output_name="mixed-test.png", image_dpi=30,
        )
        pairs = [
            ("x + 1", "2**x"),
            ("2*x - 3", "2**x - 4"),
            ("-x + 4", "3**(x - 1)"),
            ("0.5*x - 2", "(1/2)**x + 1"),
            ("2**x", "x + 1"),
            ("x + 1", "x**2 - 4*x + 3"),
            ("x**2", "-x**2 + 4"),
        ]
        for pair in pairs:
            with self.subTest(pair=pair):
                create_mixed_graph(list(pair), settings)
        self.assertEqual(savefig.call_count, len(pairs))


if __name__ == "__main__":
    unittest.main()
