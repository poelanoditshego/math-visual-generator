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
    format_coordinate,
    graph_label,
    graph_legend_is_enabled,
    intercepts_enabled,
    parse_arithmetic_expression,
    place_graph_curve_label,
    validate_polynomial_degree,
)
from models.graph_settings import GraphSettings


def create_quadratic_graph(equation: str, settings: GraphSettings) -> None:
    x, expression = parse_arithmetic_expression(
        equation,
        graph_name="Quadratic",
        example="x**2 - 4*x + 3",
    )
    polynomial = validate_polynomial_degree(expression, x, "Quadratic", 2)

    graph_function = sp.lambdify(x, expression, "numpy")
    x_values = np.linspace(settings.x_min, settings.x_max, 1000)
    y_values = np.asarray(graph_function(x_values), dtype=float)
    x_intercepts = sp.solve(expression, x)
    y_intercept = expression.subs(x, 0)
    turning_x_values = sp.solve(sp.diff(expression, x), x)

    output_folder = Path("generated_graphs")
    output_folder.mkdir(exist_ok=True)
    output_path = output_folder / settings.output_name

    _, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    labeler = PointLabeler(settings.point_label_style)
    intercept_labeler = AxisInterceptLabeler(
        settings.axis_intercept_label_style, shared=labeler
    )
    plotted_points: set[tuple[float, float]] = set()

    def plot_intercept(x_value, y_value, axis, offset):
        key = (round(float(x_value), 9), round(float(y_value), 9))
        if key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=5)
            plotted_points.add(key)
        if settings.show_point_labels:
            annotate_axis_intercept(
                ax, intercept_labeler, settings, x_value, y_value, axis, offset
            )

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

    if intercepts_enabled(settings, "x"):
        for root in x_intercepts:
            if root.is_real:
                root_value = float(root)
                if settings.x_min <= root_value <= settings.x_max:
                    plot_intercept(
                        root_value, 0, "x", settings.x_intercept_label_offset
                    )

    if intercepts_enabled(settings, "y"):
        y_intercept_value = float(y_intercept)
        if (
            settings.x_min <= 0 <= settings.x_max
            and settings.y_min <= y_intercept_value <= settings.y_max
        ):
            plot_intercept(
                0, y_intercept_value, "y", settings.y_intercept_label_offset
            )

    for turning_x in turning_x_values:
        if not turning_x.is_real:
            continue
        turning_x_value = float(turning_x)
        turning_y_value = float(expression.subs(x, turning_x))
        turning_point_is_visible = (
            settings.x_min <= turning_x_value <= settings.x_max
            and settings.y_min <= turning_y_value <= settings.y_max
        )
        if settings.show_turning_point and turning_point_is_visible:
            ax.scatter(turning_x_value, turning_y_value, zorder=5)
            if settings.show_point_labels:
                annotate_point(
                    ax,
                    labeler,
                    settings,
                    turning_x_value,
                    turning_y_value,
                    settings.turning_point_label_offset,
                )
        if settings.show_axis_of_symmetry:
            ax.axvline(
                turning_x_value,
                linestyle=":",
                label=f"Axis of symmetry: x = {format_coordinate(turning_x_value)}",
            )

    for additional_x in settings.additional_x_values:
        additional_y = float(expression.subs(x, additional_x))
        point_is_visible = (
            settings.x_min <= additional_x <= settings.x_max
            and settings.y_min <= additional_y <= settings.y_max
        )
        if point_is_visible:
            ax.scatter(additional_x, additional_y, zorder=6)
            if settings.show_additional_point_labels:
                annotate_point(
                    ax,
                    labeler,
                    settings,
                    additional_x,
                    additional_y,
                    settings.additional_point_label_offset,
                )

    ax.set_xlim(settings.x_min, settings.x_max)
    ax.set_ylim(settings.y_min, settings.y_max)
    configure_cartesian_axes(ax, settings)
    if settings.show_title:
        ax.set_title(settings.title or "Quadratic Function")

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
