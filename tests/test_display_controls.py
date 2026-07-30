import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generators.graph_helpers import (
    AxisInterceptLabeler,
    PointRegistry,
    configure_cartesian_axes,
    graph_curve_label,
    integer_ticks,
)
from generators.linear import create_linear_graph
from generators.mixed import create_mixed_graph
from models.graph_settings import GraphSettings


class DisplayControlTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def settings(self, **overrides):
        values = {
            "x_min": -4.5,
            "x_max": 4.5,
            "y_min": -4.5,
            "y_max": 6.5,
            "show_title": False,
            "show_legend": False,
            "show_graph_arrows": False,
            "output_name": "display-controls.png",
            "image_dpi": 30,
        }
        values.update(overrides)
        return GraphSettings(**values)

    def render_mixed(self, equations, **overrides):
        settings = self.settings(
            show_graph_arrows=False,
            show_turning_point=False,
            show_axis_of_symmetry=False,
            **overrides,
        )
        with patch("generators.mixed.plt.close"), patch(
            "generators.mixed.plt.savefig"
        ):
            create_mixed_graph(equations, settings)
        return plt.gca()

    @staticmethod
    def labels_at(ax, coordinate):
        return [
            text.get_text()
            for text in ax.texts
            if getattr(text, "xy", None) is not None
            and np.allclose(text.xy, coordinate, atol=1e-7)
            and text.get_text()
        ]

    @staticmethod
    def marker_count_at(ax, coordinate):
        return sum(
            1
            for collection in ax.collections
            for point in collection.get_offsets()
            if np.allclose(point, coordinate, atol=1e-7)
        )

    def test_curve_label_styles_and_mixed_names(self):
        import sympy as sp

        x = sp.Symbol("x")
        expression = 2 * x + 1
        self.assertEqual(graph_curve_label(expression, 0, "Function notation"), "$f(x)$")
        self.assertEqual(graph_curve_label(expression, 1, "Function name only"), "$g$")
        self.assertIn("f(x) =", graph_curve_label(expression, 0, "Full equation"))
        self.assertIsNone(graph_curve_label(expression, 0, "No label"))

    def test_axis_value_only_uses_the_relevant_coordinate(self):
        labeler = AxisInterceptLabeler("Axis value only")
        self.assertEqual(labeler.format_axis_label(-3, 0, "x"), "-3")
        self.assertEqual(labeler.format_axis_label(0, 6, "y"), "6")
        self.assertEqual(labeler.format_axis_label(0, 0, "x"), "0")

    def test_point_registry_merges_categories_at_one_coordinate(self):
        registry = PointRegistry(tolerance=1e-7)
        registry.add(0, 6)
        registry.add(2e-9, 6, is_graph_intersection=True)
        records = registry.records()
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_y_intercept)
        self.assertTrue(records[0].is_graph_intersection)

    def test_integer_ticks_cover_every_integer_inside_decimal_bounds(self):
        np.testing.assert_array_equal(
            integer_ticks(-4.5, 4.5), np.arange(-4, 5, dtype=float)
        )
        figure, ax = plt.subplots()
        settings = self.settings(axis_style="Central Cartesian axes")
        ax.set_xlim(settings.x_min, settings.x_max)
        ax.set_ylim(settings.y_min, settings.y_max)
        configure_cartesian_axes(ax, settings)
        figure.canvas.draw()
        np.testing.assert_array_equal(ax.get_xticks(), np.arange(-4, 5))
        labels = [label.get_text() for label in ax.get_xticklabels()]
        self.assertFalse(any(".0" in label for label in labels))

    @patch("generators.linear.plt.close")
    @patch("generators.linear.plt.show")
    @patch("generators.linear.plt.savefig")
    def test_linear_intercepts_are_independent_and_curve_label_matches_color(
        self, savefig, show, close
    ):
        settings = self.settings(
            show_x_intercepts=True,
            show_y_intercepts=False,
            graph_curve_label_style="Function notation",
        )
        create_linear_graph("2*x + 1", settings)
        ax = plt.gca()
        points = [
            tuple(point)
            for collection in ax.collections
            for point in collection.get_offsets()
        ]
        self.assertEqual(len(points), 1)
        self.assertTrue(np.allclose(points[0], (-0.5, 0)))
        curve = next(line for line in ax.lines if line.get_linewidth() == 2)
        label = next(text for text in ax.texts if text.get_text() == "$f(x)$")
        self.assertEqual(label.get_color(), curve.get_color())

    @patch("generators.mixed.plt.close")
    @patch("generators.mixed.plt.savefig")
    def test_mixed_intersection_has_priority_and_is_plotted_once(
        self, savefig, close
    ):
        settings = self.settings(
            show_x_intercepts=True,
            show_y_intercepts=True,
            show_intersection_points=True,
            graph_curve_label_style="Function notation",
        )
        create_mixed_graph(["x", "2*x"], settings)
        ax = plt.gca()
        origin_markers = [
            tuple(point)
            for collection in ax.collections
            for point in collection.get_offsets()
            if np.allclose(point, (0, 0))
        ]
        self.assertEqual(len(origin_markers), 1)
        texts = [text.get_text() for text in ax.texts]
        self.assertEqual(sum(text in {"$f(x)$", "$g(x)$"} for text in texts), 2)
        self.assertEqual(sum(text == "(0; 0)" for text in texts), 1)

    def test_y_axis_intersection_uses_axis_value_and_full_coordinate_styles(self):
        for style, expected in (
            ("Axis value only", "6"),
            ("Full coordinates", "(0; 6)"),
        ):
            with self.subTest(style=style):
                ax = self.render_mixed(
                    ["x + 6", "2*x + 6"],
                    axis_intercept_label_style=style,
                    point_label_style="Capital letter only",
                )
                self.assertEqual(self.marker_count_at(ax, (0, 6)), 1)
                self.assertEqual(self.labels_at(ax, (0, 6)), [expected])
                plt.clf()

    def test_x_axis_intersection_uses_axis_value_style(self):
        ax = self.render_mixed(
            ["x + 3", "2*x + 6"],
            axis_intercept_label_style="Axis value only",
            point_label_style="Capital letter only",
        )
        self.assertEqual(self.marker_count_at(ax, (-3, 0)), 1)
        self.assertEqual(self.labels_at(ax, (-3, 0)), ["-3"])

    def test_off_axis_intersection_uses_normal_point_style(self):
        ax = self.render_mixed(
            ["x + 3", "2*x + 1"],
            axis_intercept_label_style="Axis value only",
            point_label_style="Coordinates only",
        )
        self.assertEqual(self.marker_count_at(ax, (2, 5)), 1)
        self.assertEqual(self.labels_at(ax, (2, 5)), ["(2; 5)"])

    def test_origin_shared_point_is_plotted_and_labelled_once(self):
        ax = self.render_mixed(
            ["x", "2*x"],
            axis_intercept_label_style="Axis value only",
            point_label_style="Capital letter and coordinates",
        )
        self.assertEqual(self.marker_count_at(ax, (0, 0)), 1)
        self.assertEqual(self.labels_at(ax, (0, 0)), ["0"])

    def test_axis_intersection_visible_from_intersection_checkbox_alone(self):
        ax = self.render_mixed(
            ["x + 6", "2*x + 6"],
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_intersection_points=True,
            axis_intercept_label_style="Axis value only",
        )
        self.assertEqual(self.marker_count_at(ax, (0, 6)), 1)
        self.assertEqual(self.labels_at(ax, (0, 6)), ["6"])

    def test_axis_intersection_visible_from_y_intercept_checkbox_alone(self):
        ax = self.render_mixed(
            ["x + 6", "2*x + 6"],
            show_x_intercepts=False,
            show_y_intercepts=True,
            show_intersection_points=False,
            axis_intercept_label_style="Axis value only",
        )
        self.assertEqual(self.marker_count_at(ax, (0, 6)), 1)
        self.assertEqual(self.labels_at(ax, (0, 6)), ["6"])


if __name__ == "__main__":
    unittest.main()
