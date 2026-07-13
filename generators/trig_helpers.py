from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    parse_arithmetic_expression,
)
from models.graph_settings import GraphSettings


@dataclass(frozen=True)
class TrigParameters:
    a: float
    b: float
    c: float
    d: float

    @property
    def amplitude(self) -> float:
        return abs(self.a)

    def period(self, angle_mode: str) -> float:
        full_angle = 2 * np.pi if angle_mode == "Radians" else 360.0
        return full_angle / abs(self.b)

    @property
    def phase_shift(self) -> float:
        return -self.c / self.b


def parse_trig_expression(
    equation: str,
    trig_name: str,
    trig_function: Callable[..., sp.Expr],
) -> tuple[sp.Symbol, sp.Expr, TrigParameters]:
    display_name = trig_name.lower()
    function_name = trig_function.__name__
    x, expression = parse_arithmetic_expression(
        equation,
        graph_name=trig_name,
        example=f"3*{function_name}(2*x - 60) + 1",
        allowed_functions={function_name: trig_function},
        allowed_constants={"pi": sp.pi},
        evaluate=False,
    )
    trig_functions = [
        function
        for function in expression.atoms(sp.Function)
        if function.func == trig_function and function.args[0].has(x)
    ]
    if not trig_functions:
        raise ValueError(
            f"The expression must contain a {display_name} function involving x."
        )
    if len(trig_functions) != 1:
        raise ValueError(f"Only one transformed {display_name} function is supported.")

    function = trig_functions[0]
    placeholder = sp.Dummy(function_name)
    transformed = expression.xreplace({function: placeholder})
    if transformed.has(x):
        raise ValueError(
            f"The variable x may only appear inside the {display_name} function."
        )
    try:
        outer_polynomial = sp.Poly(transformed, placeholder)
    except sp.PolynomialError as error:
        raise ValueError(
            f"Only a scaled {display_name} function plus a constant is supported."
        ) from error
    if outer_polynomial.degree() != 1:
        raise ValueError(
            f"Only a scaled {display_name} function plus a constant is supported."
        )

    argument = sp.expand(function.args[0])
    try:
        argument_polynomial = sp.Poly(argument, x)
    except sp.PolynomialError as error:
        raise ValueError(f"The {display_name} argument must be linear in x.") from error
    if argument_polynomial.degree() != 1:
        raise ValueError(f"The {display_name} argument must be linear in x.")

    values = (
        finite_real_number(outer_polynomial.coeff_monomial(placeholder)),
        finite_real_number(argument_polynomial.coeff_monomial(x)),
        finite_real_number(argument_polynomial.coeff_monomial(1)),
        finite_real_number(outer_polynomial.coeff_monomial(1)),
    )
    if any(value is None for value in values):
        raise ValueError(f"{trig_name} parameters must be finite real constants.")
    a, b, c, d = (float(value) for value in values if value is not None)
    if abs(a) < 1e-12 or abs(b) < 1e-12:
        raise ValueError(f"{trig_name} parameters a and b must not be zero.")
    return x, expression, TrigParameters(a, b, c, d)


def phase_solutions(
    target_angle: float,
    full_angle: float,
    parameters: TrigParameters,
    x_min: float,
    x_max: float,
) -> list[float]:
    endpoint_angles = (
        parameters.b * x_min + parameters.c,
        parameters.b * x_max + parameters.c,
    )
    low_angle, high_angle = min(endpoint_angles), max(endpoint_angles)
    first = int(np.ceil((low_angle - target_angle) / full_angle - 1e-10))
    last = int(np.floor((high_angle - target_angle) / full_angle + 1e-10))
    return sorted(
        solution
        for solution in (
            (target_angle + index * full_angle - parameters.c) / parameters.b
            for index in range(first, last + 1)
        )
        if x_min - 1e-9 <= solution <= x_max + 1e-9
    )


def _unique_values(values: list[float]) -> list[float]:
    unique: dict[float, float] = {}
    for value in values:
        unique.setdefault(round(value, 9), value)
    return sorted(unique.values())


def _targets(
    trig_name: str,
    parameters: TrigParameters,
    full_angle: float,
) -> tuple[float, float]:
    quarter = full_angle / 4
    if trig_name == "Sine":
        maximum = quarter if parameters.a > 0 else 3 * quarter
        minimum = 3 * quarter if parameters.a > 0 else quarter
    else:
        maximum = 0 if parameters.a > 0 else full_angle / 2
        minimum = full_angle / 2 if parameters.a > 0 else 0
    return maximum, minimum


def _root_angles(
    trig_name: str,
    target: float,
    full_angle: float,
    angle_mode: str,
) -> tuple[float, float]:
    if trig_name == "Sine":
        primary_radians = np.arcsin(target)
        secondary_radians = np.pi - primary_radians
    else:
        primary_radians = np.arccos(target)
        secondary_radians = -primary_radians
    if angle_mode == "Degrees":
        return tuple(np.degrees((primary_radians, secondary_radians)))
    return primary_radians, secondary_radians


def create_trig_graph(
    equation: str,
    settings: GraphSettings,
    trig_name: str,
    trig_function: Callable[..., sp.Expr],
    allow_pi_labels: bool,
) -> None:
    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")
    if settings.trig_angle_mode not in {"Radians", "Degrees"}:
        raise ValueError("Angle mode must be either Radians or Degrees.")

    x, expression, parameters = parse_trig_expression(
        equation,
        trig_name,
        trig_function,
    )
    period = parameters.period(settings.trig_angle_mode)
    cycles = (settings.x_max - settings.x_min) / period
    point_count = max(1600, int(np.ceil(cycles * 240)))
    if point_count > 200_000:
        raise ValueError(
            f"The {trig_name.lower()} frequency is too high for the selected "
            "x-range to sample reliably. Reduce the frequency or narrow the range."
        )

    angle_factor = 1 if settings.trig_angle_mode == "Radians" else sp.pi / 180
    numerical_expression = (
        parameters.a
        * trig_function(angle_factor * (parameters.b * x + parameters.c))
        + parameters.d
    )
    x_values = np.linspace(settings.x_min, settings.x_max, point_count)
    raw_y_values = finite_function_values(numerical_expression, x, x_values)
    if not np.any(np.isfinite(raw_y_values)):
        raise ValueError(
            f"The {trig_name.lower()} function produced no finite real values."
        )
    y_values = raw_y_values.copy()
    visible = (
        np.isfinite(y_values)
        & (y_values >= settings.y_min)
        & (y_values <= settings.y_max)
    )
    y_values[~visible] = np.nan
    if not np.any(visible):
        raise ValueError(
            f"The {trig_name.lower()} graph has no visible real values in the graph range."
        )

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

    function_label = (
        graph_label(expression, 0, settings.graph_label_style)
        if settings.show_equation
        else None
    )
    (graph_line,) = ax.plot(x_values, y_values, linewidth=2, label=function_label)
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

    def evaluate_at(x_value: float) -> float | None:
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

    full_angle = 2 * np.pi if settings.trig_angle_mode == "Radians" else 360.0
    quarter_angle = full_angle / 4
    maximum_target, minimum_target = _targets(trig_name, parameters, full_angle)
    extreme_labels = settings.show_point_labels and settings.show_extreme_point_labels
    if settings.show_maximum_points:
        maximum_y = parameters.d + parameters.amplitude
        if settings.y_min <= maximum_y <= settings.y_max:
            for maximum_x in phase_solutions(
                maximum_target,
                full_angle,
                parameters,
                settings.x_min,
                settings.x_max,
            ):
                plot_point(
                    maximum_x,
                    maximum_y,
                    settings.turning_point_label_offset,
                    extreme_labels,
                )
    if settings.show_minimum_points:
        minimum_y = parameters.d - parameters.amplitude
        if settings.y_min <= minimum_y <= settings.y_max:
            for minimum_x in phase_solutions(
                minimum_target,
                full_angle,
                parameters,
                settings.x_min,
                settings.x_max,
            ):
                plot_point(
                    minimum_x,
                    minimum_y,
                    settings.turning_point_label_offset,
                    extreme_labels,
                )

    if settings.show_intercepts:
        target = -parameters.d / parameters.a
        if abs(target) <= 1 + 1e-12:
            target = float(np.clip(target, -1, 1))
            roots: list[float] = []
            for angle in _root_angles(
                trig_name,
                target,
                full_angle,
                settings.trig_angle_mode,
            ):
                roots.extend(
                    phase_solutions(
                        angle,
                        full_angle,
                        parameters,
                        settings.x_min,
                        settings.x_max,
                    )
                )
            for root in _unique_values(roots):
                plot_point(
                    root,
                    0.0,
                    settings.x_intercept_label_offset,
                    settings.show_point_labels,
                )
        if settings.x_min <= 0 <= settings.x_max:
            y_intercept = evaluate_at(0)
            if y_intercept is not None and settings.y_min <= y_intercept <= settings.y_max:
                plot_point(
                    0.0,
                    y_intercept,
                    settings.y_intercept_label_offset,
                    settings.show_point_labels,
                )

    show_key_points = settings.show_standard_trig_points or (
        settings.show_sine_key_points
        if trig_name == "Sine"
        else settings.show_cosine_key_points
    )
    if show_key_points:
        endpoint_angles = (
            parameters.b * settings.x_min + parameters.c,
            parameters.b * settings.x_max + parameters.c,
        )
        first = int(np.ceil(min(endpoint_angles) / quarter_angle - 1e-10))
        last = int(np.floor(max(endpoint_angles) / quarter_angle + 1e-10))
        for index in range(first, last + 1):
            key_x = (index * quarter_angle - parameters.c) / parameters.b
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
        ax.set_title(settings.title or f"{trig_name} Function")

    draw_origin_label(ax, settings)
    configure_trig_x_ticks(
        ax,
        settings.x_min,
        settings.x_max,
        settings.trig_angle_mode,
        settings.show_pi_tick_labels and allow_pi_labels,
        hide_zero=(
            settings.show_axes
            and settings.show_origin_label
            and settings.x_min <= 0 <= settings.x_max
            and settings.y_min <= 0 <= settings.y_max
        ),
        show_degree_symbols=settings.show_degree_symbols,
    )
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
