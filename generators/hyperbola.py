from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class HyperbolaParameters:
    a: float
    p: float
    q: float


def parse_hyperbola_expression(
    equation: str,
) -> tuple[sp.Symbol, sp.Expr, HyperbolaParameters]:
    """Validate and extract a, p, and q from a/(x-p)+q."""

    x, expression = parse_arithmetic_expression(
        equation,
        graph_name="Hyperbola",
        example="2/(x - 3) + 4",
    )
    terms = sp.Add.make_args(sp.expand(expression))
    denominator_terms = [term for term in terms if sp.denom(term).has(x)]

    if not denominator_terms:
        raise ValueError(
            "The expression is not a hyperbola because x does not appear "
            "in a denominator."
        )
    if len(denominator_terms) != 1:
        raise ValueError(
            "Only one constant-over-linear denominator is supported, in the "
            "form a/(x - p) + q."
        )

    fraction_term = denominator_terms[0]
    numerator, denominator = sp.fraction(fraction_term)
    constant_terms = [term for term in terms if term != fraction_term]
    if numerator.has(x) or any(term.has(x) for term in constant_terms):
        raise ValueError(
            "The numerator must be constant. Use the explicit form "
            "a/(x - p) + q."
        )

    try:
        denominator_polynomial = sp.Poly(denominator, x)
    except sp.PolynomialError as error:
        raise ValueError(
            "The denominator must be linear in x, in the form x - p."
        ) from error
    if denominator_polynomial.degree() != 1:
        raise ValueError(
            "Only a linear denominator is supported; expressions such as "
            "1/x**2 are not rectangular hyperbolas."
        )

    denominator_coefficient = denominator_polynomial.coeff_monomial(x)
    denominator_constant = denominator_polynomial.coeff_monomial(1)
    q_expression = sp.simplify(sum(constant_terms, sp.Integer(0)))
    a_value = finite_real_number(numerator / denominator_coefficient)
    p_value = finite_real_number(-denominator_constant / denominator_coefficient)
    q_value = finite_real_number(q_expression)
    if a_value is None or p_value is None or q_value is None or abs(a_value) < 1e-12:
        raise ValueError(
            "The hyperbola parameters a, p, and q must be finite real numbers, "
            "and a must not be zero."
        )

    return x, expression, HyperbolaParameters(a_value, p_value, q_value)


def create_hyperbola_graph(equation: str, settings: GraphSettings) -> None:
    """Generate a rectangular hyperbola without joining across its asymptote."""

    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")

    x, expression, parameters = parse_hyperbola_expression(equation)
    x_span = settings.x_max - settings.x_min
    asymptote_tolerance = max(1e-9, x_span * 1e-12)
    for additional_x in settings.additional_x_values:
        if np.isfinite(additional_x) and np.isclose(
            additional_x,
            parameters.p,
            atol=asymptote_tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"x = {format_coordinate(parameters.p)} is undefined because "
                "it is the vertical asymptote. Remove it from additional x-values."
            )

    gap = max(x_span / 2000, np.finfo(float).eps * max(1, abs(parameters.p)))
    x_sections: list[np.ndarray] = []
    left_end = min(settings.x_max, parameters.p - gap)
    if settings.x_min < left_end:
        x_sections.append(np.linspace(settings.x_min, left_end, 1000))
    right_start = max(settings.x_min, parameters.p + gap)
    if right_start < settings.x_max:
        x_sections.append(np.linspace(right_start, settings.x_max, 1000))

    plotted_sections: list[tuple[np.ndarray, np.ndarray]] = []
    for x_values in x_sections:
        y_values = finite_function_values(expression, x, x_values)
        visible = (
            np.isfinite(y_values)
            & (y_values >= settings.y_min)
            & (y_values <= settings.y_max)
        )
        y_values[~visible] = np.nan
        plotted_sections.append((x_values, y_values))

    if not plotted_sections or not any(
        np.any(np.isfinite(y_values)) for _, y_values in plotted_sections
    ):
        raise ValueError("The hyperbola has no visible real values in the graph range.")

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
    graph_color: str | None = None
    plotted_lines = []
    for section_index, (x_values, y_values) in enumerate(plotted_sections):
        (graph_line,) = ax.plot(
            x_values,
            y_values,
            linewidth=2,
            color=graph_color,
            label=function_label if section_index == 0 else None,
        )
        graph_color = graph_line.get_color()
        plotted_lines.append((x_values, y_values, graph_line))

    if settings.show_axes:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    if (
        settings.show_vertical_asymptote
        and settings.x_min <= parameters.p <= settings.x_max
    ):
        ax.axvline(parameters.p, linestyle="--", linewidth=1.2, color="gray")
        if settings.show_asymptote_labels:
            ax.annotate(
                f"x = {format_coordinate(parameters.p)}",
                (parameters.p, settings.y_max),
                textcoords="offset points",
                xytext=(6, -8),
                ha="left",
                va="top",
                fontsize=settings.annotation_font_size,
                bbox=annotation_box(settings),
                zorder=9,
            )
    if (
        settings.show_horizontal_asymptote
        and settings.y_min <= parameters.q <= settings.y_max
    ):
        ax.axhline(parameters.q, linestyle="--", linewidth=1.2, color="gray")
        if settings.show_asymptote_labels:
            ax.annotate(
                f"y = {format_coordinate(parameters.q)}",
                (settings.x_max, parameters.q),
                textcoords="offset points",
                xytext=(-8, 5),
                ha="right",
                va="bottom",
                fontsize=settings.annotation_font_size,
                bbox=annotation_box(settings),
                zorder=9,
            )

    def plot_point(
        x_value: float,
        y_value: float,
        offset: tuple[int, int],
        show_label: bool,
        marker: str = "o",
    ) -> None:
        point_key = (round(x_value, 9), round(y_value, 9))
        if point_key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=7, marker=marker)
            plotted_points.add(point_key)
        if show_label:
            annotate_point(ax, labeler, settings, x_value, y_value, offset)

    centre_is_visible = (
        settings.x_min <= parameters.p <= settings.x_max
        and settings.y_min <= parameters.q <= settings.y_max
    )
    if settings.show_hyperbola_centre and centre_is_visible:
        plot_point(
            parameters.p,
            parameters.q,
            settings.intersection_label_offset,
            settings.show_point_labels,
            marker="x",
        )

    if settings.show_intercepts:
        try:
            x_intercepts = sp.solve(sp.Eq(expression, 0), x)
        except (NotImplementedError, ValueError, TypeError):
            x_intercepts = []
        for root in x_intercepts:
            root_value = finite_real_number(root)
            if (
                root_value is not None
                and not np.isclose(
                    root_value,
                    parameters.p,
                    atol=asymptote_tolerance,
                    rtol=0,
                )
                and settings.x_min <= root_value <= settings.x_max
            ):
                plot_point(
                    root_value,
                    0.0,
                    settings.x_intercept_label_offset,
                    settings.show_point_labels,
                )

        if (
            settings.x_min <= 0 <= settings.x_max
            and not np.isclose(
                parameters.p,
                0,
                atol=asymptote_tolerance,
                rtol=0,
            )
        ):
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
        ax.set_title(settings.title or "Hyperbola Function")

    draw_origin_label(ax, settings)
    for x_values, y_values, graph_line in plotted_lines:
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
