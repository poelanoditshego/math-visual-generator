from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from generators.graph_helpers import (
    PointLabeler,
    annotate_point,
    annotation_box,
    configure_trig_x_ticks,
    draw_graph_end_arrows,
    draw_origin_label,
    finite_function_values,
    finite_real_number,
    format_coordinate,
    graph_label,
    graph_legend_is_enabled,
)
from generators.trig_helpers import (
    TrigParameters,
    parse_trig_expression,
    phase_solutions,
    unique_values,
)
from models.graph_settings import GraphSettings


class TangentParameters(TrigParameters):
    """Parameters for a transformed tangent function."""

    def period(self, angle_mode: str) -> float:
        full_angle = np.pi if angle_mode == "Radians" else 180.0
        return full_angle / abs(self.b)


def parse_tangent_expression(
    equation: str,
) -> tuple[sp.Symbol, sp.Expr, TangentParameters]:
    """Validate and extract a, b, c and d from a*tan(b*x+c)+d."""

    x, expression, parameters = parse_trig_expression(equation, "Tangent", sp.tan)
    return x, expression, TangentParameters(
        parameters.a,
        parameters.b,
        parameters.c,
        parameters.d,
    )


def _is_asymptote(
    x_value: float,
    parameters: TrigParameters,
    angle_mode: str,
) -> bool:
    angle = parameters.b * x_value + parameters.c
    angle_radians = np.deg2rad(angle) if angle_mode == "Degrees" else angle
    return bool(abs(np.cos(angle_radians)) < 1e-9)


def _format_asymptote(value: float, angle_mode: str) -> str:
    if angle_mode == "Degrees":
        return f"x = {format_coordinate(value)}\N{DEGREE SIGN}"

    fraction = Fraction(float(value / np.pi)).limit_denominator(16)
    numerator, denominator = fraction.numerator, fraction.denominator
    if numerator == 0:
        angle_text = "0"
    else:
        sign = "-" if numerator < 0 else ""
        magnitude = abs(numerator)
        coefficient = "" if magnitude == 1 else str(magnitude)
        if denominator == 1:
            angle_text = rf"${sign}{coefficient}\pi$"
        else:
            angle_text = rf"${sign}\frac{{{coefficient}\pi}}{{{denominator}}}$"
    return f"x = {angle_text}"


def _labelled_asymptotes(asymptotes: list[float], limit: int = 8) -> set[float]:
    if len(asymptotes) <= limit:
        return set(asymptotes)
    indices = np.linspace(0, len(asymptotes) - 1, limit, dtype=int)
    return {asymptotes[index] for index in indices}


def create_tangent_graph(equation: str, settings: GraphSettings) -> None:
    """Create a discontinuity-safe transformed tangent graph."""

    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")
    if settings.trig_angle_mode not in {"Radians", "Degrees"}:
        raise ValueError("Angle mode must be either Radians or Degrees.")

    x, expression, parameters = parse_tangent_expression(equation)
    full_angle = np.pi if settings.trig_angle_mode == "Radians" else 180.0
    period = parameters.period(settings.trig_angle_mode)
    cycles = (settings.x_max - settings.x_min) / period
    if max(1600, int(np.ceil(cycles * 240))) > 200_000:
        raise ValueError(
            "The tangent frequency is too high for the selected x-range to "
            "sample reliably. Reduce the frequency or narrow the range."
        )

    angle_factor = 1 if settings.trig_angle_mode == "Radians" else sp.pi / 180
    numerical_expression = (
        parameters.a
        * sp.tan(angle_factor * (parameters.b * x + parameters.c))
        + parameters.d
    )
    asymptote_target = full_angle / 2
    asymptotes = unique_values(
        phase_solutions(
            asymptote_target,
            full_angle,
            parameters,
            settings.x_min,
            settings.x_max,
        )
    )

    for additional_x in settings.additional_x_values:
        if (
            np.isfinite(additional_x)
            and settings.x_min <= additional_x <= settings.x_max
            and _is_asymptote(float(additional_x), parameters, settings.trig_angle_mode)
        ):
            suffix = "\N{DEGREE SIGN}" if settings.trig_angle_mode == "Degrees" else ""
            raise ValueError(
                f"The tangent function is undefined at "
                f"x = {format_coordinate(additional_x)}{suffix}."
            )

    output_directory = Path("generated_graphs")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / settings.output_name
    _, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    labeler = PointLabeler(
        settings.point_label_style,
        x_suffix=(
            "\N{DEGREE SIGN}"
            if settings.trig_angle_mode == "Degrees" and settings.show_degree_symbols
            else ""
        ),
    )
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
    if settings.show_axes:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    if settings.show_midline and settings.y_min <= parameters.d <= settings.y_max:
        ax.axhline(parameters.d, linestyle="--", linewidth=1.2, color="gray")
        if settings.show_midline_label:
            ax.annotate(
                f"y = {format_coordinate(parameters.d)}",
                (settings.x_max, parameters.d),
                textcoords="offset points",
                xytext=(-8, 5),
                ha="right",
                va="bottom",
                fontsize=settings.annotation_font_size,
                bbox=annotation_box(settings),
                zorder=9,
            )

    if settings.show_vertical_asymptote:
        labelled = _labelled_asymptotes(asymptotes)
        for asymptote in asymptotes:
            ax.axvline(asymptote, linestyle="--", linewidth=1.2, color="gray")
            if settings.show_asymptote_labels and asymptote in labelled:
                ax.annotate(
                    _format_asymptote(asymptote, settings.trig_angle_mode),
                    (asymptote, settings.y_max),
                    textcoords="offset points",
                    xytext=(4, -8),
                    ha="left",
                    va="top",
                    fontsize=settings.annotation_font_size,
                    bbox=annotation_box(settings),
                    zorder=9,
                )

    function_label = (
        graph_label(expression, 0, settings.graph_label_style)
        if settings.show_equation
        else None
    )
    boundaries = [settings.x_min, *asymptotes, settings.x_max]
    gap = max(period * 1e-5, (settings.x_max - settings.x_min) * 1e-9)
    safe_threshold = max(
        1_000_000.0,
        max(abs(settings.y_min), abs(settings.y_max)) * 10_000,
    )
    branch_data: list[tuple[np.ndarray, np.ndarray]] = []
    graph_color: str | None = None
    label_pending = function_label

    for index, (boundary_left, boundary_right) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        left = boundary_left + (gap if index > 0 else 0)
        right = boundary_right - (gap if index < len(boundaries) - 2 else 0)
        if right <= left:
            continue
        point_count = max(120, int(np.ceil((right - left) / period * 240)))
        x_values = np.linspace(left, right, point_count)
        y_values = finite_function_values(numerical_expression, x, x_values)
        y_values[np.abs(y_values) > safe_threshold] = np.nan
        visible = (
            np.isfinite(y_values)
            & (y_values >= settings.y_min)
            & (y_values <= settings.y_max)
        )
        y_values[~visible] = np.nan
        if not np.any(visible):
            continue
        (branch_line,) = ax.plot(
            x_values,
            y_values,
            linewidth=2,
            label=label_pending,
            color=graph_color,
        )
        label_pending = None
        if graph_color is None:
            graph_color = branch_line.get_color()
        branch_data.append((x_values, y_values))

    if not branch_data or graph_color is None:
        plt.close()
        raise ValueError(
            "The tangent graph has no visible real branches in the graph range."
        )

    def evaluate_at(x_value: float) -> float | None:
        if _is_asymptote(x_value, parameters, settings.trig_angle_mode):
            return None
        return finite_real_number(numerical_expression.subs(x, x_value))

    def plot_point(
        x_value: float,
        y_value: float,
        offset: tuple[int, int],
        show_label: bool,
    ) -> None:
        key = (round(x_value, 9), round(y_value, 9))
        if key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=7)
            plotted_points.add(key)
        if show_label:
            annotate_point(ax, labeler, settings, x_value, y_value, offset)

    if settings.show_intercepts:
        principal_radians = np.arctan(-parameters.d / parameters.a)
        principal = (
            np.degrees(principal_radians)
            if settings.trig_angle_mode == "Degrees"
            else principal_radians
        )
        if settings.y_min <= 0 <= settings.y_max:
            roots = phase_solutions(
                float(principal),
                full_angle,
                parameters,
                settings.x_min,
                settings.x_max,
            )
            for root in unique_values(roots):
                if not _is_asymptote(root, parameters, settings.trig_angle_mode):
                    plot_point(
                        root,
                        0.0,
                        settings.x_intercept_label_offset,
                        settings.show_point_labels,
                    )
        if settings.x_min <= 0 <= settings.x_max:
            y_intercept = evaluate_at(0)
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

    if settings.show_standard_trig_points or settings.show_tangent_key_points:
        quarter_period = period / 4
        centres = phase_solutions(
            0.0,
            full_angle,
            parameters,
            settings.x_min - quarter_period,
            settings.x_max + quarter_period,
        )
        for centre in centres:
            for key_x in (
                centre - quarter_period,
                centre,
                centre + quarter_period,
            ):
                key_y = evaluate_at(key_x)
                if (
                    key_y is not None
                    and settings.x_min <= key_x <= settings.x_max
                    and settings.y_min <= key_y <= settings.y_max
                ):
                    plot_point(
                        key_x,
                        key_y,
                        settings.intersection_label_offset,
                        settings.show_point_labels,
                    )

    for additional_x in settings.additional_x_values:
        if not np.isfinite(additional_x):
            continue
        additional_y = evaluate_at(float(additional_x))
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
        ax.set_title(settings.title or "Tangent Function")

    draw_origin_label(ax, settings)
    configure_trig_x_ticks(
        ax,
        settings.x_min,
        settings.x_max,
        settings.trig_angle_mode,
        settings.show_pi_tick_labels,
        hide_zero=(
            settings.show_axes
            and settings.show_origin_label
            and settings.x_min <= 0 <= settings.x_max
            and settings.y_min <= 0 <= settings.y_max
        ),
        show_degree_symbols=settings.show_degree_symbols,
    )
    for x_values, y_values in branch_data:
        draw_graph_end_arrows(ax, x_values, y_values, graph_color, settings)
    if graph_legend_is_enabled(settings):
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.image_dpi)
    plt.show()
    plt.close()
    print(f"Graph saved to: {output_path}")
