from dataclasses import dataclass, field
import math
import re
from numbers import Real

from constants import (
    AXIS_INTERCEPT_LABEL_STYLES,
    GRAPH_CURVE_LABEL_STYLES,
    POINT_LABEL_STYLES,
)


@dataclass(frozen=True)
class GraphRange:
    x_min: float = -10
    x_max: float = 10
    y_min: float = -10
    y_max: float = 10

    def validate(self) -> None:
        values = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("Graph range values must be finite numbers.")
        if self.x_min >= self.x_max:
            raise ValueError("Graph x_min must be smaller than x_max.")
        if self.y_min >= self.y_max:
            raise ValueError("Graph y_min must be smaller than y_max.")


@dataclass
class GraphDisplaySettings:
    show_grid: bool = True
    show_axes: bool = True
    show_equation: bool = True
    show_title: bool = True
    show_legend: bool = True
    show_border: bool = True
    show_tick_marks: bool = True
    show_tick_labels: bool = True
    use_integer_unit_ticks: bool = True
    show_x_intercepts: bool = True
    show_y_intercepts: bool = True
    show_intersection_points: bool = True
    show_point_labels: bool = True
    point_label_style: str = "Coordinates only"
    axis_intercept_label_style: str = "Full coordinates"
    graph_curve_label_style: str = "No label"
    show_origin_label: bool = True
    show_graph_arrows: bool = True
    show_axis_arrows: bool = True
    show_axis_labels: bool = True

    # Optional points placed on the graph without exposing their coordinates.
    # ``additional_point_labels`` aligns with ``additional_x_values``.
    additional_x_values: list[float] = field(default_factory=list)
    additional_point_labels: list[str] = field(default_factory=list)
    show_additional_point_labels: bool = True

    show_gradient: bool = True
    show_gradient_triangle: bool = False
    show_turning_point: bool = True
    show_axis_of_symmetry: bool = True
    show_vertical_asymptote: bool = True
    show_horizontal_asymptote: bool = True
    show_asymptote_labels: bool = True
    show_hyperbola_centre: bool = True
    show_stationary_points: bool = True
    show_inflection_point: bool = True
    trig_angle_mode: str = "Degrees"
    show_midline: bool = True
    show_maximum_points: bool = True
    show_minimum_points: bool = True
    show_circle_centre: bool = True
    show_radius: bool = False
    show_radius_label: bool = True
    show_diameter: bool = False
    show_diameter_label: bool = True

    def validate(self) -> None:
        if self.point_label_style not in POINT_LABEL_STYLES:
            raise ValueError(f"Unsupported point label style: {self.point_label_style}")
        if self.axis_intercept_label_style not in AXIS_INTERCEPT_LABEL_STYLES:
            raise ValueError(
                "Unsupported axis-intercept label style: "
                f"{self.axis_intercept_label_style}"
            )
        if self.graph_curve_label_style not in GRAPH_CURVE_LABEL_STYLES:
            raise ValueError(
                f"Unsupported graph curve label style: {self.graph_curve_label_style}"
            )
        if self.trig_angle_mode not in {"Degrees", "Radians"}:
            raise ValueError(f"Unsupported trigonometric angle mode: {self.trig_angle_mode}")
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in self.additional_x_values):
            raise ValueError("Additional x-values must be numbers.")
        if self.additional_point_labels and (
            len(self.additional_point_labels) != len(self.additional_x_values)
            or not all(isinstance(label, str) and label.strip() for label in self.additional_point_labels)
        ):
            raise ValueError("Additional point labels must match the additional x-values.")


_OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+\.png$")


def validate_output_name(output_name: object) -> None:
    if not isinstance(output_name, str) or not _OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError(
            "Output name must be a .png filename using letters, numbers, "
            "spaces, hyphens, or underscores."
        )


@dataclass
class GraphRequest:
    graph_type: str
    equation: str | None = None
    equations: list[str] | None = None
    graph_range: GraphRange = field(default_factory=GraphRange)
    display: GraphDisplaySettings = field(default_factory=GraphDisplaySettings)
    output_name: str = "graph.png"

    def validate(self) -> None:
        from generators.api import SUPPORTED_GRAPH_TYPES

        if self.graph_type not in SUPPORTED_GRAPH_TYPES:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")
        self.graph_range.validate()
        self.display.validate()
        validate_output_name(self.output_name)

        if self.graph_type == "Mixed":
            if self.equation is not None:
                raise ValueError("Mixed graph requests must use equations only.")
            if not isinstance(self.equations, list) or len(self.equations) != 2:
                raise ValueError("Mixed graph requests require exactly two equations.")
            if any(not isinstance(equation, str) or not equation.strip() for equation in self.equations):
                raise ValueError("Mixed graph requests require two non-empty equations.")
        else:
            if not isinstance(self.equation, str) or not self.equation.strip():
                raise ValueError(
                    f"An equation is required for {self.graph_type} graph requests."
                )
            if self.equations is not None:
                raise ValueError(
                    f"Only Mixed graph requests may provide multiple equations."
                )
