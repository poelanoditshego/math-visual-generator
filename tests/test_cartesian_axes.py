import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generators.graph_helpers import configure_cartesian_axes, draw_origin_label
from models.graph_settings import GraphSettings


class CartesianAxesTests(unittest.TestCase):
    def setUp(self):
        self.figure, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.figure)

    def settings(self, **overrides):
        values = {
            "axis_style": "Central Cartesian axes",
            "x_min": -10,
            "x_max": 10,
            "y_min": -10,
            "y_max": 10,
        }
        values.update(overrides)
        return GraphSettings(**values)

    def test_central_spines_ticks_arrows_labels_and_single_origin(self):
        settings = self.settings(show_border=False)
        self.ax.set_xlim(settings.x_min, settings.x_max)
        self.ax.set_ylim(settings.y_min, settings.y_max)

        configure_cartesian_axes(self.ax, settings)
        draw_origin_label(self.ax, settings)

        self.assertEqual(self.ax.spines["left"].get_position(), ("data", 0))
        self.assertEqual(self.ax.spines["bottom"].get_position(), ("data", 0))
        self.assertFalse(self.ax.spines["right"].get_visible())
        self.assertFalse(self.ax.spines["top"].get_visible())
        self.assertEqual(self.ax.xaxis.get_ticks_position(), "bottom")
        self.assertEqual(self.ax.yaxis.get_ticks_position(), "left")
        self.assertEqual([text.get_text() for text in self.ax.texts].count("0"), 1)
        self.assertIn("x", [text.get_text() for text in self.ax.texts])
        self.assertIn("y", [text.get_text() for text in self.ax.texts])
        self.assertGreaterEqual(len(self.ax.texts), 5)  # two arrows, two labels, origin

    def test_axes_outside_range_use_nearest_boundaries(self):
        settings = self.settings(x_min=2, x_max=8, y_min=3, y_max=9)
        self.ax.set_xlim(settings.x_min, settings.x_max)
        self.ax.set_ylim(settings.y_min, settings.y_max)

        configure_cartesian_axes(self.ax, settings)

        self.assertEqual(self.ax.spines["left"].get_position(), ("axes", 0))
        self.assertEqual(self.ax.spines["bottom"].get_position(), ("axes", 0))

    def test_hidden_axes_hide_central_ticks_arrows_and_labels(self):
        settings = self.settings(
            show_axes=False,
            show_border=False,
            show_axis_arrows=True,
            show_axis_labels=True,
        )

        configure_cartesian_axes(self.ax, settings)

        self.assertFalse(any(spine.get_visible() for spine in self.ax.spines.values()))
        self.assertEqual(len(self.ax.texts), 0)
        self.figure.canvas.draw()
        self.assertFalse(any(label.get_visible() for label in self.ax.get_xticklabels()))
        self.assertFalse(any(label.get_visible() for label in self.ax.get_yticklabels()))

    def test_tick_marks_can_be_hidden_without_hiding_axis_lines(self):
        settings = self.settings(show_tick_marks=False)

        configure_cartesian_axes(self.ax, settings)

        self.assertTrue(self.ax.spines["left"].get_visible())
        self.assertTrue(self.ax.spines["bottom"].get_visible())
        self.figure.canvas.draw()
        self.assertTrue(
            all(not tick.tick1line.get_visible() for tick in self.ax.xaxis.majorTicks)
        )

    def test_custom_axis_labels_are_independent_of_legend_labels(self):
        settings = self.settings(x_axis_label="f(x)", y_axis_label="g(x)")

        configure_cartesian_axes(self.ax, settings)

        labels = {text.get_text(): text.xy for text in self.ax.texts if text.get_text()}
        self.assertEqual(labels["f(x)"][1], 0)
        self.assertEqual(labels["g(x)"][0], 0)
        self.assertIsNone(self.ax.get_legend())

    def test_border_style_labels_the_horizontal_axis_x_and_vertical_axis_y(self):
        settings = self.settings(
            axis_style="Border axes",
            x_label="horizontal x",
            y_label="vertical y",
        )
        self.ax.set_xlim(settings.x_min, settings.x_max)
        self.ax.set_ylim(settings.y_min, settings.y_max)

        configure_cartesian_axes(self.ax, settings)

        labels = {text.get_text(): text.xy for text in self.ax.texts if text.get_text()}
        self.assertEqual(labels["horizontal x"][1], 0)
        self.assertEqual(labels["vertical y"][0], 0)
        self.assertEqual(self.ax.get_xlabel(), "")
        self.assertEqual(self.ax.get_ylabel(), "")


if __name__ == "__main__":
    unittest.main()
