from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from generators.graph_helpers import (
    PointLabeler,
    annotate_point,
    annotation_box,
    draw_origin_label,
    finite_real_number,
    format_coordinate,
    graph_legend_is_enabled,
)
from models.graph_settings import GraphSettings


_SAFE_CIRCLE_SIDE = re.compile(r"[0-9A-Za-z_+\-*/(),.\s]+")


@dataclass(frozen=True)
class CircleParameters:
    h: float
    k: float
    radius_squared: float

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.radius_squared))

    @property
    def diameter(self) -> float:
        return 2 * self.radius


def parse_circle_equation(
    equation: str,
) -> tuple[sp.Symbol, sp.Symbol, sp.Equality, CircleParameters]:
    """Validate a two-variable equation and extract its circle parameters."""

    if not isinstance(equation, str) or not equation.strip():
        raise ValueError("Enter a circle equation such as x**2 + y**2 = 25.")
    equals_count = equation.count("=")
    if equals_count == 0:
        raise ValueError("A circle equation must contain one equals sign.")
    if equals_count > 1:
        raise ValueError("A circle equation must contain exactly one equals sign.")

    left_text, right_text = (side.strip() for side in equation.split("=", 1))
    if not left_text or not right_text:
        raise ValueError(
            "Both sides of the circle equation must contain an expression."
        )
    for side in (left_text, right_text):
        identifiers = set(re.findall(r"[A-Za-z_]\w*", side))
        unsupported = identifiers - {"x", "y"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                "Circle equations may only use the variables x and y; "
                f"unsupported name: {names}."
            )
        if len(side) > 500 or _SAFE_CIRCLE_SIDE.fullmatch(side) is None:
            raise ValueError(
                "The circle equation contains unsupported characters or functions."
            )

    x, y = sp.symbols("x y", real=True)
    try:
        left = sp.sympify(left_text, locals={"x": x, "y": y})
        right = sp.sympify(right_text, locals={"x": x, "y": y})
    except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "The circle equation could not be understood. Use a format such as "
            "(x - 2)**2 + (y + 3)**2 = 16."
        ) from error

    if (left.free_symbols | right.free_symbols) - {x, y}:
        raise ValueError("Circle equations may only contain the variables x and y.")
    if left.atoms(sp.Function) or right.atoms(sp.Function):
        raise ValueError("Functions are not supported in a circle equation.")

    expanded = sp.expand(left - right)
    try:
        polynomial = sp.Poly(expanded, x, y)
    except sp.PolynomialError as error:
        raise ValueError(
            "The equation must be a quadratic polynomial in x and y."
        ) from error

    allowed_monomials = {(2, 0), (0, 2), (1, 0), (0, 1), (0, 0)}
    present_monomials = {monomial for monomial, _ in polynomial.terms()}
    xy_coefficient = polynomial.coeff_monomial((1, 1))
    xy_value = finite_real_number(xy_coefficient)
    if xy_value is None or abs(xy_value) > 1e-12:
        raise ValueError("Rotated conics with an x*y term are not supported.")
    if polynomial.total_degree() != 2 or not present_monomials <= allowed_monomials:
        raise ValueError("The equation is not a supported circle.")

    x_squared = polynomial.coeff_monomial((2, 0))
    y_squared = polynomial.coeff_monomial((0, 2))
    x_squared_value = finite_real_number(x_squared)
    y_squared_value = finite_real_number(y_squared)
    if (
        x_squared_value is None
        or y_squared_value is None
        or abs(x_squared_value) <= 1e-12
        or abs(y_squared_value) <= 1e-12
    ):
        raise ValueError("A circle needs non-zero x**2 and y**2 coefficients.")
    if not np.isclose(x_squared_value, y_squared_value, atol=1e-12, rtol=1e-12):
        raise ValueError("The x**2 and y**2 coefficients must be equal for a circle.")

    x_linear = polynomial.coeff_monomial((1, 0))
    y_linear = polynomial.coeff_monomial((0, 1))
    constant = polynomial.coeff_monomial((0, 0))
    h_exact = sp.simplify(-x_linear / (2 * x_squared))
    k_exact = sp.simplify(-y_linear / (2 * x_squared))
    radius_squared_exact = sp.simplify(
        h_exact**2 + k_exact**2 - constant / x_squared
    )
    values = tuple(
        finite_real_number(value)
        for value in (h_exact, k_exact, radius_squared_exact)
    )
    if any(value is None for value in values):
        raise ValueError("The circle parameters must be finite real numbers.")
    h_value, k_value, radius_squared_value = (
        float(value) for value in values if value is not None
    )
    if radius_squared_value < -1e-12:
        raise ValueError("The equation represents an imaginary circle.")
    if radius_squared_value <= 1e-12:
        raise ValueError("The circle radius must be greater than zero.")

    circle_equation = sp.Eq(left, right, evaluate=False)
    return (
        x,
        y,
        circle_equation,
        CircleParameters(h_value, k_value, radius_squared_value),
    )


def calculate_circle_cardinal_points(
    parameters: CircleParameters,
) -> list[tuple[float, float]]:
    h, k, radius = parameters.h, parameters.k, parameters.radius
    return [
        (h + radius, k),
        (h, k + radius),
        (h - radius, k),
        (h, k - radius),
    ]


def calculate_circle_intercepts(
    parameters: CircleParameters,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return the real x-axis and y-axis intercepts without tangent duplicates."""

    tolerance = 1e-10
    x_intercepts: list[tuple[float, float]] = []
    x_discriminant = parameters.radius_squared - parameters.k**2
    if x_discriminant >= -tolerance:
        offset = float(np.sqrt(max(0.0, x_discriminant)))
        x_intercepts.append((parameters.h + offset, 0.0))
        if offset > tolerance:
            x_intercepts.append((parameters.h - offset, 0.0))

    y_intercepts: list[tuple[float, float]] = []
    y_discriminant = parameters.radius_squared - parameters.h**2
    if y_discriminant >= -tolerance:
        offset = float(np.sqrt(max(0.0, y_discriminant)))
        y_intercepts.append((0.0, parameters.k + offset))
        if offset > tolerance:
            y_intercepts.append((0.0, parameters.k - offset))
    return x_intercepts, y_intercepts


def calculate_circle_angle_points(
    parameters: CircleParameters,
    angles: list[float],
) -> list[tuple[float, float]]:
    points = []
    for angle in angles:
        if not np.isfinite(angle):
            continue
        radians = np.deg2rad(angle)
        points.append(
            (
                parameters.h + parameters.radius * np.cos(radians),
                parameters.k + parameters.radius * np.sin(radians),
            )
        )
    return points


def _circle_graph_label(
    circle_equation: sp.Equality,
    style: str,
) -> str | None:
    if style == "No graph label":
        return None
    if style == "Function name only":
        return "$C$"
    equation_latex = (
        f"{sp.latex(circle_equation.lhs)} = {sp.latex(circle_equation.rhs)}"
    )
    if style == "Function equation":
        return rf"$C: {equation_latex}$"
    return rf"${equation_latex}$"


def create_circle_graph(equation: str, settings: GraphSettings) -> None:
    """Generate a circle from a validated two-variable equation."""

    ranges = (settings.x_min, settings.x_max, settings.y_min, settings.y_max)
    if not all(np.isfinite(ranges)):
        raise ValueError("Graph ranges must contain finite numbers.")
    if settings.x_min >= settings.x_max or settings.y_min >= settings.y_max:
        raise ValueError("Graph minimum values must be smaller than maximum values.")

    _, _, circle_equation, parameters = parse_circle_equation(equation)
    if any(not np.isfinite(angle) for angle in settings.additional_circle_angles):
        raise ValueError("Additional circle angles must be finite numbers.")

    theta = np.linspace(0, 2 * np.pi, 1200)
    x_values = parameters.h + parameters.radius * np.cos(theta)
    y_values = parameters.k + parameters.radius * np.sin(theta)
    visible_curve = (
        (x_values >= settings.x_min)
        & (x_values <= settings.x_max)
        & (y_values >= settings.y_min)
        & (y_values <= settings.y_max)
    )
    if not np.any(visible_curve):
        raise ValueError("The circle is not visible in the selected graph range.")

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

    circle_label = (
        _circle_graph_label(circle_equation, settings.graph_label_style)
        if settings.show_equation
        else None
    )
    (circle_line,) = ax.plot(x_values, y_values, linewidth=2, label=circle_label)
    if settings.show_axes:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    def is_visible(point: tuple[float, float]) -> bool:
        return (
            settings.x_min <= point[0] <= settings.x_max
            and settings.y_min <= point[1] <= settings.y_max
        )

    def plot_point(
        point: tuple[float, float],
        offset: tuple[int, int],
        show_label: bool,
        marker: str = "o",
    ) -> None:
        x_value, y_value = point
        key = (round(float(x_value), 9), round(float(y_value), 9))
        if key not in plotted_points:
            ax.scatter(x_value, y_value, zorder=7, marker=marker)
            plotted_points.add(key)
        if show_label:
            annotate_point(ax, labeler, settings, x_value, y_value, offset)

    centre = (parameters.h, parameters.k)
    if settings.show_circle_centre and is_visible(centre):
        plot_point(
            centre,
            settings.intersection_label_offset,
            settings.show_point_labels,
            marker="x",
        )

    cardinal_points = calculate_circle_cardinal_points(parameters)
    if settings.show_circle_cardinal_points:
        for point in cardinal_points:
            if is_visible(point):
                plot_point(
                    point,
                    settings.intersection_label_offset,
                    settings.show_point_labels,
                )

    if settings.show_intercepts:
        x_intercepts, y_intercepts = calculate_circle_intercepts(parameters)
        for point in x_intercepts:
            if is_visible(point):
                plot_point(
                    point,
                    settings.x_intercept_label_offset,
                    settings.show_point_labels,
                )
        for point in y_intercepts:
            if is_visible(point):
                plot_point(
                    point,
                    settings.y_intercept_label_offset,
                    settings.show_point_labels,
                )

    for point in calculate_circle_angle_points(
        parameters,
        settings.additional_circle_angles,
    ):
        if is_visible(point):
            plot_point(
                point,
                settings.additional_point_label_offset,
                settings.show_additional_point_labels,
            )

    radius_endpoint: tuple[float, float] | None = None
    if settings.show_radius:
        radius_preferences = (
            [cardinal_points[1], cardinal_points[3], *cardinal_points[::2]]
            if settings.show_diameter
            else cardinal_points
        )
        radius_endpoint = next(
            (point for point in radius_preferences if is_visible(point)),
            None,
        )
        if radius_endpoint is None:
            visible_indices = np.flatnonzero(visible_curve)
            if visible_indices.size:
                index = int(visible_indices[len(visible_indices) // 2])
                radius_endpoint = (float(x_values[index]), float(y_values[index]))
        if radius_endpoint is not None:
            ax.plot(
                [parameters.h, radius_endpoint[0]],
                [parameters.k, radius_endpoint[1]],
                color=circle_line.get_color(),
                linestyle="--",
                linewidth=1.3,
            )
            midpoint = (
                (parameters.h + radius_endpoint[0]) / 2,
                (parameters.k + radius_endpoint[1]) / 2,
            )
            if settings.show_radius_label and is_visible(midpoint):
                ax.annotate(
                    f"r = {format_coordinate(parameters.radius)}",
                    midpoint,
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=settings.annotation_font_size,
                    bbox=annotation_box(settings),
                    zorder=9,
                )

    if settings.show_diameter:
        opposite_pairs = (
            (cardinal_points[0], cardinal_points[2]),
            (cardinal_points[1], cardinal_points[3]),
        )
        diameter_start, diameter_end = next(
            (
                pair
                for pair in opposite_pairs
                if is_visible(pair[0]) and is_visible(pair[1])
            ),
            opposite_pairs[0],
        )
        ax.plot(
            [diameter_start[0], diameter_end[0]],
            [diameter_start[1], diameter_end[1]],
            color=circle_line.get_color(),
            linestyle=":",
            linewidth=1.3,
        )
        if settings.show_diameter_label and is_visible(centre):
            ax.annotate(
                f"d = {format_coordinate(parameters.diameter)}",
                centre,
                textcoords="offset points",
                xytext=(6, -16 if settings.show_radius_label else 6),
                fontsize=settings.annotation_font_size,
                bbox=annotation_box(settings),
                zorder=9,
            )

    ax.set_xlim(settings.x_min, settings.x_max)
    ax.set_ylim(settings.y_min, settings.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(settings.x_label)
    ax.set_ylabel(settings.y_label)
    if settings.show_title:
        ax.set_title(settings.title or "Circle")

    draw_origin_label(ax, settings)
    if graph_legend_is_enabled(settings):
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings.image_dpi)
    plt.show()
    plt.close()
    print(f"Graph saved to: {output_path}")
