from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from generators.graph_helpers import (
    PointLabeler,
    AxisInterceptLabeler,
    annotate_axis_intercept,
    annotate_point,
    configure_cartesian_axes,
    draw_graph_end_arrows,
    draw_origin_label,
    finite_function_values,
    finite_real_number,
    graph_label,
    graph_legend_is_enabled,
    intercepts_enabled,
    parse_arithmetic_expression,
    place_graph_curve_label,
    validate_polynomial_degree,
)
from models.graph_settings import GraphSettings


def parse_cubic_expression(
    equation: str,
) -> tuple[sp.Symbol, sp.Expr, sp.Poly]:
    """Parse an expression and require a polynomial of degree exactly three."""

    x, expression = parse_arithmetic_expression(
        equation,
        graph_name="Cubic",
        example="x**3 - 4*x",
    )
    expression = sp.simplify(expression)
    polynomial = validate_polynomial_degree(expression, x, "Cubic", 3)
    return x, expression, polynomial


def _stationary_point_type(
    second_derivative: sp.Expr,
    x: sp.Symbol,
    x_value: float,
) -> str:
    second_derivative_value = finite_real_number(
        second_derivative.subs(x, x_value)
    )
    if second_derivative_value is None or abs(second_derivative_value) < 1e-9:
        return "Stationary inflection"
    if second_derivative_value > 0:
        return "Minimum"
    return "Maximum"


def create_cubic_graph(equation: str, settings: GraphSettings) -> None:
    """Generate a cubic graph with its notable points and shared styling."""

    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")

    x, expression, _polynomial = parse_cubic_expression(equation)
    x_values = np.linspace(settings.x_min, settings.x_max, 1600)
    raw_y_values = finite_function_values(expression, x, x_values)
    if not np.any(np.isfinite(raw_y_values)):
        raise ValueError(
            "The cubic produced no finite real values in the selected x-range; "
            "numerical overflow may have occurred."
        )

    y_values = raw_y_values.copy()
    visible = (
        np.isfinite(y_values)
        & (y_values >= settings.y_min)
        & (y_values <= settings.y_max)
    )
    y_values[~visible] = np.nan
    if not np.any(visible):
        raise ValueError("The cubic has no visible real values in the graph range.")

    output_directory = Path("generated_graphs")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / settings.output_name

    _, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    labeler = PointLabeler(settings.point_label_style)
    intercept_labeler = AxisInterceptLabeler(
        settings.axis_intercept_label_style, shared=labeler
    )
    plotted_points: set[tuple[float, float]] = set()

    function_label = None
    if settings.show_equation:
        function_label = graph_label(expression, 0, settings.graph_label_style)
    (graph_line,) = ax.plot(
        x_values,
        y_values,
        linewidth=2,
        label=function_label,
    )

    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_point(
        x_value: float,
        y_value: float,
        offset: tuple[int, int],
        show_label: bool,
        prefix: str | None = None,
    ) -> None:
        point_key = (round(x_value, 9), round(y_value, 9))
        if point_key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=7)
            plotted_points.add(point_key)
        if show_label:
            annotate_point(
                ax,
                labeler,
                settings,
                x_value,
                y_value,
                offset,
                prefix=prefix,
            )

    def plot_intercept(x_value, y_value, axis, offset):
        point_key = (round(x_value, 9), round(y_value, 9))
        if point_key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=7)
            plotted_points.add(point_key)
        if settings.show_point_labels:
            annotate_axis_intercept(
                ax, intercept_labeler, settings, x_value, y_value, axis, offset
            )

    derivative = sp.diff(expression, x)
    second_derivative = sp.diff(expression, x, 2)
    try:
        stationary_x_values = sp.solve(sp.Eq(derivative, 0), x)
    except (NotImplementedError, ValueError, TypeError):
        stationary_x_values = []

    if settings.show_stationary_points:
        for stationary_x in stationary_x_values:
            stationary_x_value = finite_real_number(stationary_x)
            if stationary_x_value is None:
                continue
            stationary_y_value = finite_real_number(
                expression.subs(x, stationary_x)
            )
            point_is_visible = (
                stationary_y_value is not None
                and settings.x_min <= stationary_x_value <= settings.x_max
                and settings.y_min <= stationary_y_value <= settings.y_max
            )
            if point_is_visible and stationary_y_value is not None:
                prefix = None
                if settings.show_stationary_point_type:
                    prefix = _stationary_point_type(
                        second_derivative,
                        x,
                        stationary_x_value,
                    )
                plot_point(
                    stationary_x_value,
                    stationary_y_value,
                    settings.turning_point_label_offset,
                    (
                        settings.show_point_labels
                        and settings.show_stationary_point_labels
                    ),
                    prefix=prefix,
                )

    try:
        inflection_x_values = sp.solve(sp.Eq(second_derivative, 0), x)
    except (NotImplementedError, ValueError, TypeError):
        inflection_x_values = []
    if settings.show_inflection_point:
        for inflection_x in inflection_x_values:
            inflection_x_value = finite_real_number(inflection_x)
            if inflection_x_value is None:
                continue
            inflection_y_value = finite_real_number(
                expression.subs(x, inflection_x)
            )
            point_is_visible = (
                inflection_y_value is not None
                and settings.x_min <= inflection_x_value <= settings.x_max
                and settings.y_min <= inflection_y_value <= settings.y_max
            )
            if point_is_visible and inflection_y_value is not None:
                plot_point(
                    inflection_x_value,
                    inflection_y_value,
                    settings.intersection_label_offset,
                    (
                        settings.show_point_labels
                        and settings.show_inflection_point_label
                    ),
                )

    if intercepts_enabled(settings, "x"):
        try:
            x_intercepts = sp.solve(sp.Eq(expression, 0), x)
        except (NotImplementedError, ValueError, TypeError):
            x_intercepts = []
        for root in x_intercepts:
            root_value = finite_real_number(root)
            if root_value is not None and settings.x_min <= root_value <= settings.x_max:
                plot_intercept(
                    root_value, 0.0, "x", settings.x_intercept_label_offset
                )

    if intercepts_enabled(settings, "y"):
        if settings.x_min <= 0 <= settings.x_max:
            y_intercept = finite_real_number(expression.subs(x, 0))
            if (
                y_intercept is not None
                and settings.y_min <= y_intercept <= settings.y_max
            ):
                plot_intercept(
                    0.0, y_intercept, "y", settings.y_intercept_label_offset
                )

    for additional_x in settings.additional_x_values:
        if not np.isfinite(additional_x):
            continue
        additional_y = finite_real_number(expression.subs(x, additional_x))
        if (
            additional_y is not None
            and settings.x_min <= additional_x <= settings.x_max
            and settings.y_min <= additional_y <= settings.y_max
        ):
            plot_point(
                float(additional_x),
                additional_y,
                settings.additional_point_label_offset,
                settings.show_additional_point_labels,
            )

    ax.set_xlim(settings.x_min, settings.x_max)
    ax.set_ylim(settings.y_min, settings.y_max)
    configure_cartesian_axes(ax, settings)
    if settings.show_title:
        ax.set_title(settings.title or "Cubic Function")

    draw_origin_label(ax, settings)
    draw_graph_end_arrows(
        ax,
        x_values,
        y_values,
        graph_line.get_color(),
        settings,
    )
    place_graph_curve_label(
        ax, x_values, y_values, graph_line.get_color(), settings, expression
    )
    if graph_legend_is_enabled(settings):
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.image_dpi)
    plt.show()
    plt.close()
    print(f"Graph saved to: {output_path}")
