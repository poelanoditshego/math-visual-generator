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
    graph_label,
    graph_legend_is_enabled,
    intercepts_enabled,
    place_graph_curve_label,
)
from models.graph_settings import GraphSettings


def create_linear_graph(equation: str, settings: GraphSettings) -> None:
    """Generate a linear graph using the supplied customisation settings."""

    x = sp.Symbol("x")
    try:
        expression = sp.sympify(equation)
        polynomial = sp.Poly(expression, x)
    except (sp.SympifyError, sp.PolynomialError, TypeError) as error:
        raise ValueError(
            "The equation could not be understood. Use a format such as 2*x - 4."
        ) from error

    if polynomial.degree() != 1:
        raise ValueError("This generator only accepts linear expressions.")

    gradient = polynomial.coeff_monomial(x)
    y_intercept = expression.subs(x, 0)
    graph_function = sp.lambdify(x, expression, "numpy")
    x_values = np.linspace(settings.x_min, settings.x_max, 1000)
    y_values = np.asarray(graph_function(x_values), dtype=float)
    x_intercepts = sp.solve(expression, x)

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

    if settings.show_gradient:
        ax.plot(
            [],
            [],
            linestyle="",
            label=f"Gradient: m = {float(gradient):g}",
        )

    if settings.show_gradient_triangle:
        gradient_value = float(gradient)
        start_x = 0.0
        start_y = float(y_intercept)
        run = 1.0
        rise = gradient_value * run
        end_x = start_x + run
        end_y = start_y + rise
        triangle_is_visible = (
            settings.x_min <= start_x <= settings.x_max
            and settings.x_min <= end_x <= settings.x_max
            and settings.y_min <= start_y <= settings.y_max
            and settings.y_min <= end_y <= settings.y_max
        )
        if triangle_is_visible:
            ax.plot([start_x, end_x], [start_y, start_y], linestyle="--")
            ax.plot([end_x, end_x], [start_y, end_y], linestyle="--")
            if settings.show_point_labels:
                ax.annotate(
                    f"Run = {run:g}",
                    ((start_x + end_x) / 2, start_y),
                    textcoords="offset points",
                    xytext=(0, -18),
                    ha="center",
                )
                ax.annotate(
                    f"Rise = {rise:g}",
                    (end_x, (start_y + end_y) / 2),
                    textcoords="offset points",
                    xytext=(8, 0),
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
        ax.set_title(settings.title or "Linear Function")

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
