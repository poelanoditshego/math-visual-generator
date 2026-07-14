from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from generators.exponential import horizontal_asymptote
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
    is_supported_exponential,
    parse_arithmetic_expression,
)
from models.graph_settings import GraphSettings


SUPPORTED_TYPES = "Linear, Quadratic or Exponential"


def detect_graph_type(expression: sp.Expr, x: sp.Symbol) -> str:
    """Classify one supported Mixed expression using its SymPy structure."""

    if is_supported_exponential(expression, x):
        return "Exponential"
    if expression.is_polynomial(x):
        degree = sp.Poly(expression, x).degree()
        if degree == 1:
            return "Linear"
        if degree == 2:
            return "Quadratic"
    raise ValueError(f"The expression is not a supported {SUPPORTED_TYPES} expression.")


def parse_mixed_expression(equation: str, equation_number: int) -> tuple[sp.Symbol, sp.Expr, str]:
    """Parse and classify one equation while identifying validation errors clearly."""

    try:
        x, expression = parse_arithmetic_expression(
            equation,
            graph_name=f"Equation {equation_number}",
            example="x + 1 or 2**x",
        )
        graph_type = detect_graph_type(expression, x)
    except ValueError as error:
        message = str(error)
        if message.startswith(f"Equation {equation_number}"):
            raise
        raise ValueError(f"Equation {equation_number}: {message}") from error
    return x, expression, graph_type


def _unique_numbers(values: list[float], tolerance: float = 1e-7) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _bisect_root(function, left: float, right: float) -> float | None:
    """Refine a bracketed continuous root without requiring SciPy."""

    try:
        left_value = float(function(left))
        right_value = float(function(right))
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return None
    for _ in range(70):
        middle = (left + right) / 2
        try:
            middle_value = float(function(middle))
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(middle_value):
            return None
        if abs(middle_value) < 1e-11 or right - left < 1e-10:
            return middle
        if np.signbit(left_value) != np.signbit(middle_value):
            right, right_value = middle, middle_value
        else:
            left, left_value = middle, middle_value
    return (left + right) / 2


def find_real_roots(
    expression: sp.Expr,
    x: sp.Symbol,
    x_min: float,
    x_max: float,
) -> list[float]:
    """Find symbolic, crossing, and practical tangent roots in a finite interval."""

    roots: list[float] = []
    try:
        symbolic_roots = sp.solve(sp.Eq(expression, 0), x)
    except (NotImplementedError, ValueError, TypeError):
        symbolic_roots = []
    for root in symbolic_roots:
        value = finite_real_number(root)
        if value is not None and x_min - 1e-8 <= value <= x_max + 1e-8:
            roots.append(float(np.clip(value, x_min, x_max)))

    sample_x = np.linspace(x_min, x_max, 4001)
    sample_y = finite_function_values(expression, x, sample_x)
    function = sp.lambdify(x, expression, modules=["numpy"])
    finite = np.isfinite(sample_y)
    for index in range(len(sample_x) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        if sample_y[index] == 0:
            roots.append(float(sample_x[index]))
        elif np.signbit(sample_y[index]) != np.signbit(sample_y[index + 1]):
            root = _bisect_root(function, float(sample_x[index]), float(sample_x[index + 1]))
            if root is not None:
                roots.append(root)

    # nsolve near local minima of |f| catches common tangent-style intersections.
    absolute = np.abs(sample_y)
    scale = max(1.0, float(np.nanmax(np.minimum(absolute[finite], 1e6)))) if np.any(finite) else 1.0
    candidate_indices = np.flatnonzero(
        finite[1:-1]
        & (absolute[1:-1] <= absolute[:-2])
        & (absolute[1:-1] <= absolute[2:])
        & (absolute[1:-1] < max(1e-4, scale * 1e-5))
    ) + 1
    for index in candidate_indices:
        try:
            root = sp.nsolve(expression, x, float(sample_x[index]), verify=False)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        value = finite_real_number(root)
        if value is None or not (x_min - 1e-7 <= value <= x_max + 1e-7):
            continue
        residual = finite_real_number(expression.subs(x, value))
        if residual is not None and abs(residual) < 1e-6:
            roots.append(float(np.clip(value, x_min, x_max)))
    return _unique_numbers(roots)


def create_mixed_graph(equations: list[str], settings: GraphSettings) -> None:
    """Plot two automatically detected linear, quadratic, or exponential graphs."""

    if len(equations) != 2:
        raise ValueError("Enter exactly two equations for a mixed graph.")
    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")

    parsed = [parse_mixed_expression(equation, index + 1) for index, equation in enumerate(equations)]
    x = parsed[0][0]
    expressions = [item[1] for item in parsed]
    graph_types = [item[2] for item in parsed]
    x_values = np.linspace(settings.x_min, settings.x_max, 2000)

    output_directory = Path("generated_graphs")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / settings.output_name
    _, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    labeler = PointLabeler(settings.point_label_style)
    plotted_points: set[tuple[float, float]] = set()
    plotted_graphs: list[tuple[np.ndarray, object]] = []
    plotted_asymptotes: set[float] = set()

    for spine in ax.spines.values():
        spine.set_visible(settings.show_border)
    ax.tick_params(
        axis="both", which="both",
        bottom=settings.show_tick_marks, top=settings.show_tick_marks,
        left=settings.show_tick_marks, right=settings.show_tick_marks,
        labelbottom=settings.show_tick_labels, labeltop=False,
        labelleft=settings.show_tick_labels, labelright=False,
    )

    def plot_point(x_value: float, y_value: float, offset: tuple[int, int], show_label: bool) -> None:
        key = (round(x_value, 8), round(y_value, 8))
        if key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=8)
            plotted_points.add(key)
        if show_label:
            annotate_point(ax, labeler, settings, x_value, y_value, offset)

    for function_index, (expression, graph_type) in enumerate(zip(expressions, graph_types)):
        y_values = finite_function_values(expression, x, x_values)
        visible = (
            np.isfinite(y_values)
            & (y_values >= settings.y_min)
            & (y_values <= settings.y_max)
        )
        if not np.any(visible):
            raise ValueError(f"Equation {function_index + 1} has no visible real values in the graph range.")
        plotted_y = y_values.copy()
        plotted_y[~visible] = np.nan
        label = graph_label(expression, function_index, settings.graph_label_style) if settings.show_equation else None
        (line,) = ax.plot(x_values, plotted_y, linewidth=2, label=label)
        plotted_graphs.append((plotted_y, line))

        if settings.show_intercepts:
            for root in find_real_roots(expression, x, settings.x_min, settings.x_max):
                if settings.y_min <= 0 <= settings.y_max:
                    plot_point(root, 0.0, settings.x_intercept_label_offset, settings.show_point_labels)
            if settings.x_min <= 0 <= settings.x_max:
                y_intercept = finite_real_number(expression.subs(x, 0))
                if y_intercept is not None and settings.y_min <= y_intercept <= settings.y_max:
                    plot_point(0.0, y_intercept, settings.y_intercept_label_offset, settings.show_point_labels)

        if graph_type == "Quadratic":
            polynomial = sp.Poly(expression, x)
            a, b = polynomial.all_coeffs()[:2]
            turning_x = finite_real_number(-b / (2 * a))
            turning_y = finite_real_number(expression.subs(x, turning_x)) if turning_x is not None else None
            if turning_x is not None and turning_y is not None:
                visible_turning = settings.x_min <= turning_x <= settings.x_max and settings.y_min <= turning_y <= settings.y_max
                if settings.show_turning_point and visible_turning:
                    plot_point(turning_x, turning_y, settings.turning_point_label_offset, settings.show_point_labels)
                if settings.show_axis_of_symmetry and settings.x_min <= turning_x <= settings.x_max:
                    ax.axvline(turning_x, linestyle=":", linewidth=1, color=line.get_color(), alpha=0.7)

        if graph_type == "Exponential" and settings.show_horizontal_asymptote:
            asymptote = horizontal_asymptote(expression, x)
            asymptote_key = round(asymptote, 9) if asymptote is not None else None
            if (
                asymptote is not None
                and settings.y_min <= asymptote <= settings.y_max
                and asymptote_key not in plotted_asymptotes
            ):
                plotted_asymptotes.add(asymptote_key)
                ax.axhline(asymptote, linestyle="--", linewidth=1.2, color=line.get_color(), alpha=0.7)
                if settings.horizontal_asymptote_label:
                    ax.annotate(
                        f"y = {format_coordinate(asymptote)}", (settings.x_max, asymptote),
                        textcoords="offset points", xytext=(-8, 5), ha="right", va="bottom",
                        fontsize=settings.annotation_font_size, bbox=annotation_box(settings), zorder=9,
                    )

        for additional_x in settings.additional_x_values:
            if not np.isfinite(additional_x) or not settings.x_min <= additional_x <= settings.x_max:
                continue
            additional_y = finite_real_number(expression.subs(x, additional_x))
            if additional_y is not None and settings.y_min <= additional_y <= settings.y_max:
                plot_point(float(additional_x), additional_y, settings.additional_point_label_offset, settings.show_additional_point_labels)

    if settings.show_intersection_points:
        difference = expressions[0] - expressions[1]
        for intersection_x in find_real_roots(difference, x, settings.x_min, settings.x_max):
            intersection_y = finite_real_number(expressions[0].subs(x, intersection_x))
            if intersection_y is not None and settings.y_min <= intersection_y <= settings.y_max:
                plot_point(intersection_x, intersection_y, settings.intersection_label_offset, settings.show_point_labels)

    if settings.show_axes:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(settings.x_min, settings.x_max)
    ax.set_ylim(settings.y_min, settings.y_max)
    ax.set_xlabel(settings.x_label)
    ax.set_ylabel(settings.y_label)
    if settings.show_title:
        ax.set_title(settings.title or "Mixed Functions")

    draw_origin_label(ax, settings)
    for y_values, line in plotted_graphs:
        draw_graph_end_arrows(ax, x_values, y_values, line.get_color(), settings)
    if graph_legend_is_enabled(settings):
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.image_dpi)
    plt.close()
