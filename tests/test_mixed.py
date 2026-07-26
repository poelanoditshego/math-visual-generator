import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
from generators.hyperbola import extract_hyperbola_parameters
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
            "1/x": "Hyperbola",
            "2/(x - 1) + 2": "Hyperbola",
        }
        for text, graph_type in expected.items():
            with self.subTest(text=text):
                self.assertEqual(detect_graph_type(self.expression(text), self.x), graph_type)
        self.assertEqual(parse_mixed_expression("y = 2**x", 1)[2], "Exponential")
        self.assertEqual(parse_mixed_expression("y = x + 1", 2)[2], "Linear")

    def test_rejects_unsupported_expressions_with_equation_number(self):
        for text in ("sin(x)", "log(x)", "1/x**2", "tan(x)", "x**3"):
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

    def test_hyperbola_parameters_and_line_intersections(self):
        cases = [
            ("2*x + 1", "1/x", (1.0, 0.0, 0.0), [-1.0, 0.5]),
            ("-x + 4", "2/(x - 1) + 2", (2.0, 1.0, 2.0), []),
        ]
        for linear, hyperbola, expected_parameters, expected_roots in cases:
            with self.subTest(linear=linear, hyperbola=hyperbola):
                expression = self.expression(hyperbola)
                parameters = extract_hyperbola_parameters(expression, self.x)
                self.assertEqual(
                    (parameters.a, parameters.p, parameters.q),
                    expected_parameters,
                )
                roots = find_real_roots(
                    self.expression(linear) - expression, self.x, -10, 10
                )
                self.assertEqual(len(roots), len(expected_roots))
                for actual, wanted in zip(roots, expected_roots):
                    self.assertAlmostEqual(actual, wanted, places=6)

    @patch("generators.mixed.plt.close")
    @patch("generators.mixed.plt.savefig")
    def test_quadratic_exponential_both_orders_render_features(self, savefig, close):
        settings = GraphSettings(
            x_min=-5, x_max=5, y_min=-2, y_max=20,
            show_turning_point=True,
            show_horizontal_asymptote=True,
            show_intersection_points=True,
            output_name="mixed-quadratic-exponential.png",
            image_dpi=30,
        )
        expected_intersections = [-0.76666469596, 2.0, 4.0]
        for pair in (("x**2", "2**x"), ("2**x", "x**2")):
            with self.subTest(pair=pair):
                create_mixed_graph(list(pair), settings)
                ax = plt.gca()
                graph_lines = [line for line in ax.lines if line.get_linewidth() == 2]
                self.assertEqual(len(graph_lines), 2)
                horizontal_asymptotes = [
                    line for line in ax.lines
                    if line.get_linestyle() == "--"
                    and np.allclose(line.get_ydata(), 0)
                ]
                self.assertEqual(len(horizontal_asymptotes), 1)

                points = [
                    tuple(point)
                    for collection in ax.collections
                    for point in collection.get_offsets()
                ]
                self.assertTrue(any(np.allclose(point, (0, 0)) for point in points))
                for intersection_x in expected_intersections:
                    self.assertTrue(any(
                        np.allclose(
                            point,
                            (intersection_x, intersection_x**2),
                            atol=1e-6,
                        )
                        for point in points
                    ))
                self.assertTrue(
                    any(getattr(annotation, "arrow_patch", None) is not None for annotation in ax.texts)
                )
                plt.clf()

    @patch("generators.mixed.plt.close")
    @patch("generators.mixed.plt.savefig")
    def test_linear_hyperbola_both_orders_render_split_branches(self, savefig, close):
        settings = GraphSettings(
            x_min=-10, x_max=10, y_min=-10, y_max=10,
            show_graph_arrows=True, output_name="mixed-hyperbola.png", image_dpi=30,
        )
        pairs = [
            ("2*x + 1", "1/x"),
            ("1/x", "2*x + 1"),
            ("-x + 4", "2/(x - 1) + 2"),
            ("2/(x - 1) + 2", "-x + 4"),
        ]
        for pair in pairs:
            with self.subTest(pair=pair):
                create_mixed_graph(list(pair), settings)
                ax = plt.gca()
                graph_lines = [line for line in ax.lines if line.get_linewidth() == 2]
                self.assertEqual(len(graph_lines), 3)
                hyperbola_lines = graph_lines[:2] if "1/" in pair[0] or "/(" in pair[0] else graph_lines[1:]
                self.assertEqual(len(hyperbola_lines), 2)
                asymptote = 0 if "1/x" in pair else 1
                self.assertTrue(all(
                    np.all(line.get_xdata() < asymptote)
                    or np.all(line.get_xdata() > asymptote)
                    for line in hyperbola_lines
                ))
                self.assertEqual(
                    sum(not line.get_label().startswith("_") for line in hyperbola_lines),
                    1,
                )
                dashed_vertical = [
                    line for line in ax.lines
                    if line.get_linestyle() == "--"
                    and np.allclose(line.get_xdata(), asymptote)
                ]
                self.assertEqual(len(dashed_vertical), 1)
                self.assertTrue(
                    any(getattr(annotation, "arrow_patch", None) is not None for annotation in ax.texts)
                )
                plt.clf()

    @patch("generators.mixed.plt.savefig")
    def test_additional_x_at_hyperbola_asymptote_warns_and_skips(self, savefig):
        settings = GraphSettings(
            additional_x_values=[1], output_name="mixed-warning.png", image_dpi=30
        )
        with self.assertWarnsRegex(UserWarning, "vertical asymptote"):
            create_mixed_graph(["-x + 4", "2/(x - 1) + 2"], settings)

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
            ("x**2", "2**x"),
            ("2**x", "x**2"),
            ("2*x + 1", "1/x"),
            ("1/x", "2*x + 1"),
            ("-x + 4", "2/(x - 1) + 2"),
            ("2/(x - 1) + 2", "-x + 4"),
        ]
        for pair in pairs:
            with self.subTest(pair=pair):
                create_mixed_graph(list(pair), settings)
        self.assertEqual(savefig.call_count, len(pairs))


if __name__ == "__main__":
    unittest.main()
