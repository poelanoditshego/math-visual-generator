import os
from pathlib import Path
import unittest

import matplotlib

matplotlib.use("Agg")

from generators.api import (
    generate_graph,
    generate_graph_from_request,
    graph_request_to_settings,
)
from models.graph_artifact import GraphArtifact
from models.graph_request import GraphDisplaySettings, GraphRange, GraphRequest
from models.graph_settings import GraphSettings


class GraphRequestTests(unittest.TestCase):
    def setUp(self):
        self.created_paths: list[Path] = []

    def tearDown(self):
        for path in self.created_paths:
            path.unlink(missing_ok=True)

    def test_linear_request_converts_and_generates(self):
        request = GraphRequest(
            graph_type="Linear",
            equation="2*x + 1",
            graph_range=GraphRange(x_min=-5, x_max=5, y_min=-5, y_max=10),
            output_name="request-test.png",
        )
        settings = graph_request_to_settings(request)
        self.assertEqual((settings.x_min, settings.x_max), (-5, 5))
        self.assertEqual((settings.y_min, settings.y_max), (-5, 10))

        artifact = generate_graph_from_request(request)
        self.created_paths.append(Path(artifact.image_path))
        self.assertIsInstance(artifact, GraphArtifact)
        self.assertEqual(artifact.graph_type, "Linear")
        self.assertTrue(os.path.isfile(artifact.image_path))
        self.assertGreater(os.path.getsize(artifact.image_path), 0)

    def test_quadratic_request_can_hide_turning_point(self):
        request = GraphRequest(
            graph_type="Quadratic",
            equation="-(x - 1)**2 + 4",
            display=GraphDisplaySettings(show_turning_point=False),
        )
        settings = graph_request_to_settings(request)
        self.assertFalse(settings.show_turning_point)

    def test_valid_mixed_and_sine_requests(self):
        mixed = GraphRequest(graph_type="Mixed", equations=["2*x + 1", "1/x"])
        mixed.validate()
        sine = GraphRequest(
            graph_type="Sine",
            equation="2*sin(x - 30) + 1",
            display=GraphDisplaySettings(trig_angle_mode="Degrees"),
        )
        sine.validate()

    def test_mixed_rejects_both_equation_channels(self):
        with self.assertRaisesRegex(ValueError, "equations only"):
            GraphRequest(
                graph_type="Mixed",
                equation="x",
                equations=["x", "x**2"],
            ).validate()

    def test_non_string_equation_is_rejected_as_value_error(self):
        with self.assertRaisesRegex(ValueError, "An equation is required"):
            GraphRequest(graph_type="Linear", equation=42).validate()

        with self.assertRaisesRegex(ValueError, "An equation is required"):
            generate_graph("Linear", GraphSettings(), equation=42)

    def test_mixed_rejects_empty_and_non_string_equations(self):
        for equations in (["x", ""], ["x", 3], [None, "x"]):
            with self.subTest(equations=equations), self.assertRaisesRegex(
                ValueError, "non-empty equations"
            ):
                GraphRequest(graph_type="Mixed", equations=equations).validate()
            with self.subTest(api_equations=equations), self.assertRaisesRegex(
                ValueError, "non-empty equations"
            ):
                generate_graph("Mixed", GraphSettings(), equations=equations)

    def test_invalid_ranges_are_rejected(self):
        for graph_range in (
            GraphRange(x_min=2, x_max=2),
            GraphRange(y_min=3, y_max=1),
        ):
            with self.subTest(graph_range=graph_range), self.assertRaisesRegex(
                ValueError, "smaller"
            ):
                GraphRequest(graph_type="Linear", equation="x", graph_range=graph_range).validate()

        for graph_range in (
            GraphRange(x_min=float("nan")),
            GraphRange(x_max=float("inf")),
            GraphRange(y_min=float("-inf")),
            GraphRange(x_min=True),
            GraphRange(y_max=False),
        ):
            with self.subTest(graph_range=graph_range), self.assertRaisesRegex(
                ValueError, "finite numbers"
            ):
                GraphRequest(graph_type="Linear", equation="x", graph_range=graph_range).validate()

    def test_invalid_request_structure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported graph type"):
            GraphRequest(graph_type="Piecewise", equation="x").validate()
        with self.assertRaisesRegex(ValueError, "An equation is required"):
            GraphRequest(graph_type="Linear").validate()
        for equations in (["x"], ["x", "x**2", "2**x"]):
            with self.subTest(equations=equations), self.assertRaisesRegex(
                ValueError, "exactly two equations"
            ):
                GraphRequest(graph_type="Mixed", equations=equations).validate()

    def test_unsafe_output_names_are_rejected(self):
        for output_name in ("../graph.png", "folder/graph.png", "", "graph.jpg"):
            with self.subTest(output_name=output_name), self.assertRaisesRegex(
                ValueError, "Output name"
            ):
                GraphRequest(
                    graph_type="Linear",
                    equation="x",
                    output_name=output_name,
                ).validate()

        for output_name in ("../direct.png", "folder/direct.png", "direct.jpg"):
            with self.subTest(output_name=output_name), self.assertRaisesRegex(
                ValueError, "Output name"
            ):
                generate_graph(
                    "Linear",
                    GraphSettings(output_name=output_name),
                    equation="x",
                )

    def test_asymptote_label_mapping_controls_exponential_labels(self):
        settings = graph_request_to_settings(
            GraphRequest(
                graph_type="Exponential",
                equation="2**x",
                display=GraphDisplaySettings(show_asymptote_labels=False),
            )
        )
        self.assertFalse(settings.show_asymptote_labels)
        self.assertFalse(settings.horizontal_asymptote_label)

    def test_all_exposed_display_fields_are_mapped(self):
        display = GraphDisplaySettings(
            show_grid=False,
            show_axes=False,
            show_border=False,
            show_tick_marks=False,
            show_tick_labels=False,
            use_integer_unit_ticks=False,
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_intersection_points=False,
            show_point_labels=False,
            point_label_style="No label",
            axis_intercept_label_style="No label",
            graph_curve_label_style="Full equation",
            show_origin_label=False,
            show_graph_arrows=False,
            show_axis_arrows=False,
            show_axis_labels=False,
            show_gradient=False,
            show_gradient_triangle=True,
            show_turning_point=False,
            show_axis_of_symmetry=False,
            show_vertical_asymptote=False,
            show_horizontal_asymptote=False,
            show_asymptote_labels=False,
            show_hyperbola_centre=False,
            show_stationary_points=False,
            show_inflection_point=False,
            trig_angle_mode="Radians",
            show_midline=False,
            show_maximum_points=False,
            show_minimum_points=False,
            show_circle_centre=False,
            show_radius=True,
            show_radius_label=False,
            show_diameter=True,
            show_diameter_label=False,
        )
        settings = graph_request_to_settings(
            GraphRequest(graph_type="Linear", equation="x", display=display)
        )
        expected = {
            field_name: value
            for field_name, value in vars(display).items()
        }
        expected["horizontal_asymptote_label"] = display.show_asymptote_labels
        for field_name, value in expected.items():
            with self.subTest(field_name=field_name):
                self.assertEqual(getattr(settings, field_name), value)

    def test_invalid_display_enum_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "angle mode"):
            GraphRequest(
                graph_type="Sine",
                equation="sin(x)",
                display=GraphDisplaySettings(trig_angle_mode="Gradians"),
            ).validate()
        with self.assertRaisesRegex(ValueError, "point label style"):
            GraphRequest(
                graph_type="Linear",
                equation="x",
                display=GraphDisplaySettings(point_label_style="Unknown"),
            ).validate()


if __name__ == "__main__":
    unittest.main()
