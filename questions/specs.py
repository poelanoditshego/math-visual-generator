"""Question-type specifications shared by generation and presentation layers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from models.graph_request import GraphDisplaySettings


@dataclass(frozen=True)
class QuestionSpec:
    """Configuration for one learner-facing question type."""

    question_type: str
    family: str
    question_template: str
    answer_type: str
    display_overrides: Mapping[str, object]
    ai_visible_information: tuple[str, ...]
    ai_hidden_information: tuple[str, ...]
    fingerprint_fields: tuple[str, ...] = ("gradient", "y_intercept")
    requires_graph: bool = True

    def build_display_settings(self) -> GraphDisplaySettings:
        """Create a fresh display policy so callers may add dynamic point data."""
        display = GraphDisplaySettings(
            show_title=False,
            show_legend=False,
            show_equation=False,
            show_grid=False,
            show_border=False,
            show_tick_labels=False,
            show_tick_marks=False,
            show_axes=True,
            show_axis_arrows=True,
            show_axis_labels=True,
            show_origin_label=True,
        )
        for name, value in self.display_overrides.items():
            setattr(display, name, value)
        return display


def _policy(**overrides: object) -> Mapping[str, object]:
    return MappingProxyType(overrides)


LINEAR_QUESTION_SPECS: dict[str, QuestionSpec] = {
    "x_intercept": QuestionSpec(
        question_type="x_intercept",
        family="linear",
        question_template="{equation_prefix}Determine the x-intercept of the graph.",
        answer_type="coordinate",
        display_overrides=_policy(show_x_intercepts=False, show_y_intercepts=True),
        ai_visible_information=("equation", "y-intercept"),
        ai_hidden_information=("x-intercept",),
    ),
    "y_intercept": QuestionSpec(
        question_type="y_intercept",
        family="linear",
        question_template="{equation_prefix}Determine the y-intercept of the graph.",
        answer_type="coordinate",
        display_overrides=_policy(show_y_intercepts=False, show_x_intercepts=True),
        ai_visible_information=("equation", "x-intercept"),
        ai_hidden_information=("y-intercept",),
    ),
    "gradient": QuestionSpec(
        question_type="gradient",
        family="linear",
        question_template="{equation_prefix}Determine the gradient of the graph.",
        answer_type="number",
        display_overrides=_policy(
            show_gradient=False,
            show_gradient_triangle=False,
            show_x_intercepts=True,
            show_y_intercepts=True,
        ),
        ai_visible_information=("equation", "x-intercept", "y-intercept"),
        ai_hidden_information=("gradient",),
    ),
    "determine_equation": QuestionSpec(
        question_type="determine_equation",
        family="linear",
        question_template="Determine the equation of the linear graph shown below.",
        answer_type="equation",
        display_overrides=_policy(
            show_x_intercepts=True,
            show_y_intercepts=True,
            show_gradient=False,
            show_gradient_triangle=False,
        ),
        ai_visible_information=("x-intercept", "y-intercept"),
        ai_hidden_information=("equation", "gradient"),
    ),
    "equation_from_two_points": QuestionSpec(
        question_type="equation_from_two_points",
        family="linear",
        question_template=(
            "Points A{point_a} and B{point_b} lie on a straight line. "
            "Determine the equation of the line passing through A and B."
        ),
        answer_type="equation",
        display_overrides=_policy(
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_gradient=False,
            show_gradient_triangle=False,
            show_additional_point_labels=True,
            show_grid=True,
            show_tick_labels=True,
            show_tick_marks=True,
        ),
        ai_visible_information=("point A", "point B"),
        ai_hidden_information=("equation", "gradient", "y-intercept"),
        fingerprint_fields=("canonical_point_pair",),
    ),
    "equation_from_gradient_and_point": QuestionSpec(
        question_type="equation_from_gradient_and_point",
        family="linear",
        question_template=(
            "A straight line has a gradient of {gradient} and passes through "
            "the point A{point_a}. Determine the equation of the line."
        ),
        answer_type="equation",
        display_overrides=_policy(
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_gradient=False,
            show_gradient_triangle=False,
            show_additional_point_labels=True,
            show_grid=True,
            show_tick_labels=True,
            show_tick_marks=True,
        ),
        ai_visible_information=("gradient", "point A"),
        ai_hidden_information=("equation", "y-intercept"),
        fingerprint_fields=("gradient", "point_a"),
    ),
    "find_f_of_x": QuestionSpec(
        question_type="find_f_of_x",
        family="linear",
        question_template=(
            "The graph of f(x) = {equation} is shown below. Determine f({input_x})."
        ),
        answer_type="number",
        display_overrides=_policy(
            show_x_intercepts=True,
            show_y_intercepts=True,
            show_gradient=False,
            show_gradient_triangle=False,
        ),
        ai_visible_information=("equation", "input_x"),
        ai_hidden_information=("f(x) value",),
        fingerprint_fields=("gradient", "y_intercept", "input_x"),
    ),
    "find_x_given_y": QuestionSpec(
        question_type="find_x_given_y",
        family="linear",
        question_template=(
            "The graph of f(x) = {equation} is shown below. "
            "Determine the value of x if f(x) = {target_y}."
        ),
        answer_type="number",
        display_overrides=_policy(
            show_x_intercepts=True,
            show_y_intercepts=True,
            show_gradient=False,
            show_gradient_triangle=False,
        ),
        ai_visible_information=("equation", "target_y"),
        ai_hidden_information=("x value",),
        fingerprint_fields=("gradient", "y_intercept", "target_y"),
    ),
    "read_coordinate": QuestionSpec(
        question_type="read_coordinate",
        family="linear",
        question_template="Write down the coordinates of point A.",
        answer_type="coordinate",
        display_overrides=_policy(
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_gradient=False,
            show_gradient_triangle=False,
            show_additional_point_labels=True,
            show_grid=True,
            show_tick_labels=True,
            show_tick_marks=True,
        ),
        ai_visible_information=("point label", "graph"),
        ai_hidden_information=("coordinates",),
        fingerprint_fields=("gradient", "y_intercept", "selected_point"),
    ),
    "increasing_or_decreasing": QuestionSpec(
        question_type="increasing_or_decreasing",
        family="linear",
        question_template=(
            "State whether the function shown below is increasing or decreasing."
        ),
        answer_type="choice",
        display_overrides=_policy(
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_gradient=False,
            show_gradient_triangle=False,
        ),
        ai_visible_information=("graph",),
        ai_hidden_information=("gradient", "equation"),
    ),
    "intersection_of_two_lines": QuestionSpec(
        question_type="intersection_of_two_lines",
        family="linear",
        question_template=(
            "The graphs of f and g are shown below.\n\n"
            "Determine the coordinates of the point of intersection of f and g."
        ),
        answer_type="coordinate",
        display_overrides=_policy(
            show_x_intercepts=False,
            show_y_intercepts=False,
            show_intersection_points=False,
            show_point_labels=False,
            show_grid=True,
            show_tick_labels=True,
            show_tick_marks=True,
            show_gradient=False,
            show_gradient_triangle=False,
            graph_curve_label_style="Function name only",
        ),
        ai_visible_information=("graphs f and g",),
        ai_hidden_information=("intersection coordinates",),
        fingerprint_fields=("canonical_line_pair", "intersection_point"),
    ),
}

QUESTION_SPECS: Mapping[tuple[str, str], QuestionSpec] = MappingProxyType(
    {(spec.family, spec.question_type): spec for spec in LINEAR_QUESTION_SPECS.values()}
)


def get_question_spec(question_type: str, *, family: str = "linear") -> QuestionSpec:
    """Return a registered specification or reject an unsupported family/type pair."""
    try:
        return QUESTION_SPECS[(family, question_type)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported {family} question type: {question_type}"
        ) from error


def get_family_question_types(family: str) -> tuple[str, ...]:
    """Return supported types in stable registration order for a family."""
    return tuple(
        spec.question_type for spec in QUESTION_SPECS.values() if spec.family == family
    )
