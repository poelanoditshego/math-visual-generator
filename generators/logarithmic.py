from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from generators.graph_helpers import (
    PointLabeler,
    annotate_point,
    annotation_box,
    draw_graph_end_arrows,
    draw_origin_label,
    finite_function_values,
    finite_real_number,
    format_coordinate,
    graph_label,
    graph_legend_is_enabled,
    parse_arithmetic_expression,
)
from models.graph_settings import GraphSettings


def parse_logarithmic_expression(
    equation: str,
) -> tuple[sp.Symbol, sp.Expr, sp.Expr, float]:
    """Validate one transformed logarithm and return its domain boundary."""

    x, expression = parse_arithmetic_expression(
        equation,
        graph_name="Logarithmic",
        example="log(x - 2, 2) + 1",
        allowed_functions={"log": sp.log},
    )
    if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
        raise ValueError(
            "The logarithm base must be a positive real number other than 1."
        )

    logarithms = [
        function
        for function in expression.atoms(sp.Function)
        if function.func == sp.log and function.args[0].has(x)
    ]
    if not logarithms:
        raise ValueError("The expression must contain a logarithm involving x.")
    if len(logarithms) != 1:
        raise ValueError(
            "This generator supports one logarithm with one linear argument."
        )

    variable_logarithm = logarithms[0]
    logarithm_placeholder = sp.Dummy("logarithm")
    transformed_expression = expression.xreplace(
        {variable_logarithm: logarithm_placeholder}
    )
    if transformed_expression.has(x):
        raise ValueError("The variable x may only appear inside the logarithm.")
    try:
        transformed_polynomial = sp.Poly(
            transformed_expression,
            logarithm_placeholder,
        )
    except sp.PolynomialError as error:
        raise ValueError(
            "Only a scaled logarithm plus a constant is supported."
        ) from error
    if transformed_polynomial.degree() != 1:
        raise ValueError("Only a scaled logarithm plus a constant is supported.")
    scale = finite_real_number(
        transformed_polynomial.coeff_monomial(logarithm_placeholder)
    )
    shift = finite_real_number(transformed_polynomial.coeff_monomial(1))
    if scale is None or shift is None or abs(scale) < 1e-12:
        raise ValueError(
            "The logarithm scale and vertical shift must be finite real numbers."
        )

    argument = sp.simplify(variable_logarithm.args[0])
    try:
        argument_polynomial = sp.Poly(argument, x)
    except sp.PolynomialError as error:
        raise ValueError("The logarithm argument must be linear in x.") from error
    if argument_polynomial.degree() != 1:
        raise ValueError("The logarithm argument must be linear in x.")

    argument_coefficient = finite_real_number(
        argument_polynomial.coeff_monomial(x)
    )
    argument_constant = finite_real_number(
        argument_polynomial.coeff_monomial(1)
    )
    if (
        argument_coefficient is None
        or argument_constant is None
        or abs(argument_coefficient) < 1e-12
    ):
        raise ValueError("The logarithm argument must have real linear coefficients.")

    vertical_asymptote = -argument_constant / argument_coefficient
    return x, expression, argument, vertical_asymptote


def create_logarithmic_graph(equation: str, settings: GraphSettings) -> None:
    """Generate a logarithmic graph over only its valid real domain."""

    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")

    x, expression, argument, vertical_asymptote = parse_logarithmic_expression(
        equation
    )
    x_span = settings.x_max - settings.x_min
    gap = max(
        x_span / 10000,
        np.finfo(float).eps * max(1, abs(vertical_asymptote)),
    )

    def argument_value(x_value: float) -> float | None:
        return finite_real_number(argument.subs(x, x_value))

    def is_in_domain(x_value: float) -> bool:
        value = argument_value(x_value)
        return value is not None and value > 0

    for additional_x in settings.additional_x_values:
        if np.isfinite(additional_x) and not is_in_domain(float(additional_x)):
            raise ValueError(
                f"x = {format_coordinate(additional_x)} is outside the "
                "logarithm's domain because its argument is not positive."
            )

    x_values = np.linspace(settings.x_min, settings.x_max, 2000)
    argument_coefficient = finite_real_number(sp.diff(argument, x))
    if argument_coefficient is not None:
        domain_direction = 1 if argument_coefficient > 0 else -1
        maximum_distance = max(
            gap,
            abs(settings.x_min - vertical_asymptote),
            abs(settings.x_max - vertical_asymptote),
        )
        near_asymptote = vertical_asymptote + domain_direction * np.geomspace(
            gap,
            maximum_distance,
            700,
        )
        near_asymptote = near_asymptote[
            (near_asymptote >= settings.x_min)
            & (near_asymptote <= settings.x_max)
        ]
        x_values = np.unique(np.concatenate((x_values, near_asymptote)))

    argument_values = finite_function_values(argument, x, x_values)
    domain_mask = np.isfinite(argument_values) & (argument_values > 0)
    if not np.any(domain_mask):
        raise ValueError(
            "No part of the selected x-range belongs to the logarithm's domain."
        )

    y_values = finite_function_values(expression, x, x_values)
    visible = (
        domain_mask
        & np.isfinite(y_values)
        & (y_values >= settings.y_min)
        & (y_values <= settings.y_max)
    )
    y_values[~visible] = np.nan
    if not np.any(visible):
        raise ValueError("The logarithm has no visible real values in the graph range.")

    output_directory = Path("generated_graphs")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / settings.output_name

    _, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    labeler = PointLabeler(settings.point_label_style)
    plotted_points: set[tuple[float, float]] = set()

    for spine in ax.spines.values():
        spine.set_visible(settings.show_border)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=settings.show_tick_marks,
        top=settings.show_tick_marks,
        left=settings.show_tick_marks,
        right=settings.show_tick_marks,
        labelbottom=settings.show_tick_labels,
        labeltop=False,
        labelleft=settings.show_tick_labels,
        labelright=False,
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

    if settings.show_axes:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    if (
        settings.show_vertical_asymptote
        and settings.x_min <= vertical_asymptote <= settings.x_max
    ):
        ax.axvline(
            vertical_asymptote,
            linestyle="--",
            linewidth=1.2,
            color="gray",
        )
        if settings.show_asymptote_labels:
            ax.annotate(
                f"x = {format_coordinate(vertical_asymptote)}",
                (vertical_asymptote, settings.y_max),
                textcoords="offset points",
                xytext=(6, -8),
                ha="left",
                va="top",
                fontsize=settings.annotation_font_size,
                bbox=annotation_box(settings),
                zorder=9,
            )

    def plot_point(
        x_value: float,
        y_value: float,
        offset: tuple[int, int],
        show_label: bool,
    ) -> None:
        point_key = (round(x_value, 9), round(y_value, 9))
        if point_key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=7)
            plotted_points.add(point_key)
        if show_label:
            annotate_point(ax, labeler, settings, x_value, y_value, offset)

    if settings.show_intercepts:
        try:
            x_intercepts = sp.solve(sp.Eq(expression, 0), x)
        except (NotImplementedError, ValueError, TypeError):
            x_intercepts = []
        for root in x_intercepts:
            root_value = finite_real_number(root)
            if (
                root_value is not None
                and is_in_domain(root_value)
                and settings.x_min <= root_value <= settings.x_max
            ):
                plot_point(
                    root_value,
                    0.0,
                    settings.x_intercept_label_offset,
                    settings.show_point_labels,
                )

        if settings.x_min <= 0 <= settings.x_max and is_in_domain(0):
            y_intercept = finite_real_number(expression.subs(x, 0))
            if (
                y_intercept is not None
                and settings.y_min <= y_intercept <= settings.y_max
            ):
                plot_point(
                    0.0,
                    y_intercept,
                    settings.y_intercept_label_offset,
                    settings.show_point_labels,
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
    ax.set_xlabel(settings.x_label)
    ax.set_ylabel(settings.y_label)
    if settings.show_title:
        ax.set_title(settings.title or "Logarithmic Function")

    draw_origin_label(ax, settings)
    draw_graph_end_arrows(
        ax,
        x_values,
        y_values,
        graph_line.get_color(),
        settings,
    )
    if graph_legend_is_enabled(settings):
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.image_dpi)
    plt.show()
    plt.close()
    print(f"Graph saved to: {output_path}")
