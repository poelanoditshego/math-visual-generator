from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from generators.graph_helpers import (
    PointLabeler,
    annotate_point,
    draw_graph_end_arrows,
    draw_origin_label,
    format_coordinate,
    graph_label,
    graph_legend_is_enabled,
)
from models.graph_settings import GraphSettings


def create_mixed_graph(equations: list[str], settings: GraphSettings) -> None:
    x = sp.Symbol("x")
    if len(equations) < 2:
        raise ValueError("Enter at least two equations for a mixed graph.")

    expressions: list[sp.Expr] = []
    for equation in equations:
        try:
            expression = sp.sympify(equation)
            polynomial = sp.Poly(expression, x)
        except (sp.SympifyError, sp.PolynomialError, TypeError) as error:
            raise ValueError(f"Invalid equation: {equation}") from error

        if expression.free_symbols and expression.free_symbols != {x}:
            raise ValueError("Equations may only contain the variable x.")
        if polynomial.degree() not in {1, 2}:
            raise ValueError(
                "Mixed graphs currently support only linear and quadratic equations."
            )
        expressions.append(expression)

    x_values = np.linspace(settings.x_min, settings.x_max, 1000)
    _, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    labeler = PointLabeler(settings.point_label_style)
    plotted_graphs: list[tuple[np.ndarray, object]] = []

    for spine in ax.spines.values():
        spine.set_visible(settings.show_border)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=settings.show_tick_marks,
        top=False,
        left=settings.show_tick_marks,
        right=False,
        labelbottom=settings.show_tick_labels,
        labeltop=False,
        labelleft=settings.show_tick_labels,
        labelright=False,
    )

    for function_index, expression in enumerate(expressions):
        polynomial = sp.Poly(expression, x)
        degree = polynomial.degree()
        function = sp.lambdify(x, expression, modules=["numpy"])
        y_values = np.asarray(function(x_values), dtype=float)
        function_label = None
        if settings.show_equation:
            function_label = graph_label(
                expression,
                function_index,
                settings.graph_label_style,
            )
        (graph_line,) = ax.plot(
            x_values,
            y_values,
            linewidth=2,
            label=function_label,
        )
        plotted_graphs.append((y_values, graph_line))

        if settings.show_intercepts:
            for x_intercept in sp.solve(sp.Eq(expression, 0), x):
                if x_intercept.is_real:
                    x_intercept_value = float(x_intercept)
                    if settings.x_min <= x_intercept_value <= settings.x_max:
                        ax.scatter(x_intercept_value, 0, zorder=7)
                        if settings.show_point_labels:
                            annotate_point(
                                ax,
                                labeler,
                                settings,
                                x_intercept_value,
                                0,
                                settings.x_intercept_label_offset,
                            )

            y_intercept_value = float(expression.subs(x, 0))
            if (
                settings.x_min <= 0 <= settings.x_max
                and settings.y_min <= y_intercept_value <= settings.y_max
            ):
                ax.scatter(0, y_intercept_value, zorder=7)
                if settings.show_point_labels:
                    annotate_point(
                        ax,
                        labeler,
                        settings,
                        0,
                        y_intercept_value,
                        settings.y_intercept_label_offset,
                    )

        if degree == 2:
            a, b = (float(value) for value in polynomial.all_coeffs()[:2])
            turning_x = -b / (2 * a)
            turning_y = float(expression.subs(x, turning_x))
            turning_point_is_visible = (
                settings.x_min <= turning_x <= settings.x_max
                and settings.y_min <= turning_y <= settings.y_max
            )
            if settings.show_turning_point and turning_point_is_visible:
                ax.scatter(turning_x, turning_y, zorder=8)
                if settings.show_point_labels:
                    annotate_point(
                        ax,
                        labeler,
                        settings,
                        turning_x,
                        turning_y,
                        settings.turning_point_label_offset,
                    )
            if settings.show_axis_of_symmetry:
                ax.axvline(
                    x=turning_x,
                    linestyle="--",
                    linewidth=1,
                    label=f"$x = {format_coordinate(turning_x)}$",
                )

        for additional_x in settings.additional_x_values:
            additional_y = float(expression.subs(x, additional_x))
            point_is_visible = (
                settings.x_min <= additional_x <= settings.x_max
                and settings.y_min <= additional_y <= settings.y_max
            )
            if point_is_visible:
                ax.scatter(additional_x, additional_y, zorder=8)
                if settings.show_additional_point_labels:
                    annotate_point(
                        ax,
                        labeler,
                        settings,
                        additional_x,
                        additional_y,
                        settings.additional_point_label_offset,
                    )

    if settings.show_intersection_points and len(expressions) == 2:
        intersection_x_values = sp.solve(
            sp.Eq(expressions[0], expressions[1]),
            x,
        )
        for intersection_x in intersection_x_values:
            if not intersection_x.is_real:
                continue
            intersection_x_value = float(intersection_x)
            intersection_y_value = float(expressions[0].subs(x, intersection_x))
            point_is_visible = (
                settings.x_min <= intersection_x_value <= settings.x_max
                and settings.y_min <= intersection_y_value <= settings.y_max
            )
            if point_is_visible:
                ax.scatter(intersection_x_value, intersection_y_value, zorder=8)
                if settings.show_point_labels:
                    annotate_point(
                        ax,
                        labeler,
                        settings,
                        intersection_x_value,
                        intersection_y_value,
                        settings.intersection_label_offset,
                    )

    if settings.show_axes:
        ax.axhline(y=0, linewidth=1)
        ax.axvline(x=0, linewidth=1)
    if settings.show_grid:
        ax.grid(True)

    ax.set_xlim(settings.x_min, settings.x_max)
    ax.set_ylim(settings.y_min, settings.y_max)
    ax.set_xlabel(settings.x_label)
    ax.set_ylabel(settings.y_label)
    if settings.show_title:
        ax.set_title(settings.title or "Mixed Functions")

    draw_origin_label(ax, settings)
    for y_values, graph_line in plotted_graphs:
        draw_graph_end_arrows(
            ax,
            x_values,
            y_values,
            graph_line.get_color(),
            settings,
        )
    if graph_legend_is_enabled(settings):
        ax.legend()

    output_directory = Path("generated_graphs")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / settings.output_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.image_dpi)
    plt.close()
