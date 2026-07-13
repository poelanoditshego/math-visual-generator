from __future__ import annotations

import re
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
)
from models.graph_settings import GraphSettings


_SAFE_EXPRESSION = re.compile(r"[0-9x+\-*/().\s]+")


def parse_exponential_expression(equation: str) -> tuple[sp.Symbol, sp.Expr]:
    """Parse and validate a real constant-base exponential expression."""

    if not isinstance(equation, str) or not equation.strip():
        raise ValueError("Enter an exponential expression such as 2**x.")
    identifiers = set(re.findall(r"[A-Za-z_]\w*", equation))
    unsupported_identifiers = identifiers - {"x"}
    if unsupported_identifiers:
        names = ", ".join(sorted(unsupported_identifiers))
        raise ValueError(
            "Exponential expressions may only use the variable x; "
            f"unsupported name: {names}."
        )
    if len(equation) > 500 or _SAFE_EXPRESSION.fullmatch(equation) is None:
        raise ValueError(
            "The expression contains unsupported characters or functions. "
            "Use numbers, x, parentheses, and arithmetic operators only."
        )

    x = sp.Symbol("x", real=True)
    try:
        expression = sp.sympify(equation, locals={"x": x})
    except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "The equation could not be understood. Use a format such as 2**x + 3."
        ) from error

    if expression.free_symbols - {x}:
        raise ValueError("Exponential expressions may only contain the variable x.")
    if expression.atoms(sp.Function):
        raise ValueError("Logarithmic and trigonometric functions are not supported.")

    exponential_powers: list[sp.Pow] = []
    for power in expression.atoms(sp.Pow):
        if not power.exp.has(x) or power.base.has(x):
            continue
        base_value = finite_real_number(power.base)
        if base_value is not None and base_value > 0 and base_value != 1:
            exponential_powers.append(power)

    if not exponential_powers:
        raise ValueError(
            "The expression is not exponential. It must contain x in the "
            "exponent of a positive constant base, for example 2**x."
        )

    replacements = {
        power: sp.Dummy(f"exponential_{index}")
        for index, power in enumerate(exponential_powers)
    }
    if expression.xreplace(replacements).has(x):
        raise ValueError(
            "The variable x may only appear inside an exponential exponent."
        )

    return x, expression


def horizontal_asymptote(
    expression: sp.Expr,
    x: sp.Symbol,
) -> float | None:
    """Return a finite horizontal end-limit when SymPy can determine one."""

    exponential_form = expression.xreplace(
        {
            power: sp.exp(
                power.exp * sp.log(power.base),
                evaluate=False,
            )
            for power in expression.atoms(sp.Pow)
            if power.exp.has(x) and not power.base.has(x)
        }
    )
    for direction in (sp.oo, -sp.oo):
        try:
            limit = sp.limit(exponential_form, x, direction)
        except (NotImplementedError, ValueError, TypeError):
            continue
        value = finite_real_number(limit)
        if value is not None:
            return value
    return None


def create_exponential_graph(equation: str, settings: GraphSettings) -> None:
    """Generate an exponential graph using the shared graph settings."""

    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")

    x, expression = parse_exponential_expression(equation)
    x_values = np.linspace(settings.x_min, settings.x_max, 1600)
    y_values = finite_function_values(expression, x, x_values)
    finite_count = int(np.count_nonzero(np.isfinite(y_values)))
    if finite_count == 0:
        raise ValueError(
            "The expression produced no finite real values in the selected x-range; "
            "numerical overflow may have occurred."
        )

    plotted_y_values = y_values.copy()
    visible = (
        np.isfinite(plotted_y_values)
        & (plotted_y_values >= settings.y_min)
        & (plotted_y_values <= settings.y_max)
    )
    plotted_y_values[~visible] = np.nan
    if not np.any(visible):
        raise ValueError("The function has no visible real values in the graph range.")

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
        plotted_y_values,
        linewidth=2,
        label=function_label,
    )

    if settings.show_axes:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    asymptote = horizontal_asymptote(expression, x)
    if (
        settings.show_horizontal_asymptote
        and asymptote is not None
        and settings.y_min <= asymptote <= settings.y_max
    ):
        ax.axhline(asymptote, linestyle="--", linewidth=1.2, color="gray")
        if settings.horizontal_asymptote_label:
            ax.annotate(
                f"y = {format_coordinate(asymptote)}",
                (settings.x_max, asymptote),
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
            if root_value is not None and settings.x_min <= root_value <= settings.x_max:
                plot_point(
                    root_value,
                    0.0,
                    settings.x_intercept_label_offset,
                    settings.show_point_labels,
                )

        if settings.x_min <= 0 <= settings.x_max:
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
        ax.set_title(settings.title or "Exponential Function")

    draw_origin_label(ax, settings)
    draw_graph_end_arrows(
        ax,
        x_values,
        plotted_y_values,
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
