from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from string import ascii_uppercase

import numpy as np
import sympy as sp
from matplotlib.axes import Axes
from matplotlib.ticker import FixedFormatter, FuncFormatter

from constants import (
    AXIS_INTERCEPT_LABEL_STYLES,
    AXIS_STYLES,
    GRAPH_CURVE_LABEL_STYLES,
    GRAPH_LABEL_STYLES,
    POINT_LABEL_STYLES,
)
from models.graph_settings import GraphSettings

_SAFE_EXPRESSION = re.compile(r"[0-9A-Za-z_+\-*/(),.\s]+")


def parse_arithmetic_expression(
    equation: str,
    graph_name: str,
    example: str,
    allowed_functions: dict[str, object] | None = None,
    allowed_constants: dict[str, object] | None = None,
    evaluate: bool = True,
) -> tuple[sp.Symbol, sp.Expr]:
    """Safely parse a one-variable arithmetic expression for a generator."""

    if not isinstance(equation, str) or not equation.strip():
        raise ValueError(f"Enter a {graph_name.lower()} expression such as {example}.")

    equation = equation.strip()
    if "=" in equation:
        parts = equation.split("=")
        if len(parts) != 2 or parts[0].strip().lower() != "y" or not parts[1].strip():
            raise ValueError("Equations must be entered as an expression or as y = expression.")
        equation = parts[1].strip()

    functions = allowed_functions or {}
    constants = allowed_constants or {}
    identifiers = set(re.findall(r"[A-Za-z_]\w*", equation))
    function_names = set(re.findall(r"([A-Za-z_]\w*)\s*\(", equation))
    unsupported_functions = function_names - set(functions)
    if unsupported_functions:
        names = ", ".join(sorted(unsupported_functions))
        raise ValueError(f"Unsupported function '{names}' in equation.")

    unsupported_identifiers = identifiers - {"x", *functions, *constants}
    if unsupported_identifiers:
        names = ", ".join(sorted(unsupported_identifiers))
        raise ValueError(f"Unsupported variable '{names}'. Only x is allowed.")
    if (
        len(equation) > 500
        or _SAFE_EXPRESSION.fullmatch(equation) is None
    ):
        raise ValueError(
            "The expression contains unsupported characters or functions. "
            "Use numbers, x, parentheses, and arithmetic operators only."
        )

    x = sp.Symbol("x", real=True)
    try:
        expression = sp.sympify(
            equation,
            locals={"x": x, **functions, **constants},
            evaluate=evaluate,
        )
    except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"The equation could not be understood. Use a format such as {example}."
        ) from error

    if expression.free_symbols - {x}:
        raise ValueError(f"{graph_name} expressions may only contain the variable x.")
    unsupported_functions = {
        function.func
        for function in expression.atoms(sp.Function)
        if function.func not in functions.values()
    }
    if unsupported_functions:
        raise ValueError("Named functions are not supported by this graph generator.")

    return x, expression


def validate_polynomial_degree(
    expression: sp.Expr,
    x: sp.Symbol,
    graph_name: str,
    degree: int,
) -> sp.Poly:
    """Require a finite real polynomial of one exact degree in x."""

    try:
        polynomial = sp.Poly(expression, x)
    except sp.PolynomialError as error:
        raise ValueError(f"The equation must be a {graph_name.lower()} function in x.") from error

    if polynomial.degree() != degree:
        raise ValueError(f"The equation must be a {graph_name.lower()} function in x.")

    if any(finite_real_number(coefficient) is None for coefficient in polynomial.all_coeffs()):
        raise ValueError(
            f"The {graph_name.lower()} coefficients must be finite real numbers."
        )
    return polynomial


def supported_exponential_powers(
    expression: sp.Expr,
    x: sp.Symbol,
) -> list[sp.Pow]:
    """Return the valid constant-base powers that make an expression exponential."""

    powers: list[sp.Pow] = []
    for power in expression.atoms(sp.Pow):
        if not power.exp.has(x) or power.base.has(x):
            continue
        base_value = finite_real_number(power.base)
        if base_value is not None and base_value > 0 and base_value != 1:
            powers.append(power)
    return powers


def is_supported_exponential(expression: sp.Expr, x: sp.Symbol) -> bool:
    """Whether x occurs only in supported positive constant-base exponents."""

    powers = supported_exponential_powers(expression, x)
    if not powers or expression.is_polynomial(x):
        return False
    replacements = {
        power: sp.Dummy(f"exponential_{index}")
        for index, power in enumerate(powers)
    }
    return not expression.xreplace(replacements).has(x)


def configure_trig_x_ticks(
    ax: Axes,
    x_min: float,
    x_max: float,
    angle_mode: str,
    show_pi_labels: bool,
    hide_zero: bool = False,
    show_degree_symbols: bool = True,
) -> None:
    """Apply readable radian or degree ticks without overcrowding the axis."""

    if angle_mode == "Radians" and not show_pi_labels:
        return

    span = x_max - x_min
    if angle_mode == "Radians":
        step = np.pi / 2
        while span / step > 10:
            step *= 2
    else:
        degree_steps = (30.0, 45.0, 60.0, 90.0, 180.0, 360.0)
        step = next(
            (candidate for candidate in degree_steps if span / candidate <= 10),
            degree_steps[-1],
        )
        while span / step > 10:
            step *= 2

    first_index = int(np.ceil(x_min / step - 1e-12))
    last_index = int(np.floor(x_max / step + 1e-12))
    ticks = np.arange(first_index, last_index + 1, dtype=float) * step
    if angle_mode == "Degrees":
        labels = [
            "" if hide_zero and abs(value) < 1e-9
            else (
                rf"${format_coordinate(value)}^\circ$"
                if show_degree_symbols
                else format_coordinate(value)
            )
            for value in ticks
        ]
    else:
        labels = []
        for value in ticks:
            fraction = Fraction(float(value / np.pi)).limit_denominator(16)
            numerator = fraction.numerator
            denominator = fraction.denominator
            if numerator == 0:
                labels.append("" if hide_zero else "$0$")
                continue
            sign = "-" if numerator < 0 else ""
            magnitude = abs(numerator)
            coefficient = "" if magnitude == 1 else str(magnitude)
            if denominator == 1:
                labels.append(rf"${sign}{coefficient}\pi$")
            else:
                numerator_text = rf"{coefficient}\pi"
                labels.append(
                    rf"${sign}\frac{{{numerator_text}}}{{{denominator}}}$"
                )
    ax.set_xticks(ticks, labels)


def format_coordinate(value: float) -> str:
    """Format a coordinate compactly while avoiding floating-point noise."""

    value = float(value)
    if abs(value) < 1e-9:
        value = 0.0
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def finite_function_values(
    expression: sp.Expr,
    symbol: sp.Symbol,
    x_values: np.ndarray,
) -> np.ndarray:
    """Evaluate an expression without allowing invalid sections to be joined."""

    function = sp.lambdify(symbol, expression, modules=["numpy"])
    with np.errstate(all="ignore"):
        raw_values = function(x_values)

    values = np.asarray(raw_values)
    if values.ndim == 0:
        values = np.full(np.asarray(x_values).shape, values)

    if np.iscomplexobj(values):
        nearly_real = np.isclose(values.imag, 0, atol=1e-10)
        values = np.where(nearly_real, values.real, np.nan)

    try:
        values = values.astype(float)
    except (TypeError, ValueError, OverflowError):
        return np.full(np.asarray(x_values).shape, np.nan, dtype=float)

    values[~np.isfinite(values)] = np.nan
    return values


def finite_real_number(value: sp.Expr | float) -> float | None:
    """Convert one symbolic value to a finite real float, or return none."""

    try:
        numeric_value = complex(sp.N(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if abs(numeric_value.imag) > 1e-10 or not np.isfinite(numeric_value.real):
        return None
    return float(numeric_value.real)


def _coordinate_key(x_value: float, y_value: float) -> tuple[float, float]:
    return (round(float(x_value), 9), round(float(y_value), 9))


@dataclass
class PointRecord:
    """All known categories and display metadata for one coordinate."""

    x: float
    y: float
    is_x_intercept: bool = False
    is_y_intercept: bool = False
    is_graph_intersection: bool = False
    is_other_special_point: bool = False
    other_label_enabled: bool = False
    other_offset: tuple[int, int] | None = None


class PointRegistry:
    """Merge duplicate coordinates without losing their point categories."""

    def __init__(self, tolerance: float = 1e-7) -> None:
        self.tolerance = tolerance
        self._records: list[PointRecord] = []

    def add(
        self,
        x_value: float,
        y_value: float,
        *,
        is_graph_intersection: bool = False,
        is_other_special_point: bool = False,
        other_label_enabled: bool = False,
        other_offset: tuple[int, int] | None = None,
    ) -> PointRecord:
        """Add or merge a point, classifying axes from its coordinates."""

        is_x_intercept = abs(float(y_value)) < self.tolerance
        is_y_intercept = abs(float(x_value)) < self.tolerance
        record = next(
            (
                existing
                for existing in self._records
                if abs(existing.x - float(x_value)) < self.tolerance
                and abs(existing.y - float(y_value)) < self.tolerance
            ),
            None,
        )
        if record is None:
            record = PointRecord(
                x=0.0 if is_y_intercept else float(x_value),
                y=0.0 if is_x_intercept else float(y_value),
            )
            self._records.append(record)
        record.is_x_intercept |= is_x_intercept
        record.is_y_intercept |= is_y_intercept
        record.is_graph_intersection |= is_graph_intersection
        record.is_other_special_point |= is_other_special_point
        record.other_label_enabled |= other_label_enabled
        if record.other_offset is None and other_offset is not None:
            record.other_offset = other_offset
        return record

    def records(self) -> list[PointRecord]:
        return list(self._records)


def _capital_name(index: int) -> str:
    """Return spreadsheet-style capital names: A..Z, AA..AZ, and so on."""

    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, len(ascii_uppercase))
        name = ascii_uppercase[remainder] + name
    return name


class PointLabeler:
    """Assign stable point names and prevent duplicate coordinate annotations."""

    def __init__(
        self,
        style: str,
        x_suffix: str = "",
        shared: PointLabeler | None = None,
    ) -> None:
        self.style = style
        self.x_suffix = x_suffix
        self._letters = shared._letters if shared is not None else {}
        self._annotated = shared._annotated if shared is not None else set()

    def format_label(self, x_value: float, y_value: float) -> str | None:
        if self.style == "No label":
            return None

        coordinates = (
            f"({format_coordinate(x_value)}{self.x_suffix}; "
            f"{format_coordinate(y_value)})"
        )
        if self.style == "Coordinates only":
            return coordinates

        key = _coordinate_key(x_value, y_value)
        letter = self._letters.setdefault(key, _capital_name(len(self._letters)))
        if self.style == "Capital letter only":
            return letter
        if self.style == "Capital letter and coordinates":
            return f"{letter}{coordinates}"

        return coordinates

    def mark_annotated(self, x_value: float, y_value: float) -> bool:
        """Return false when this coordinate has already received an annotation."""

        key = _coordinate_key(x_value, y_value)
        if key in self._annotated:
            return False
        self._annotated.add(key)
        return True


class AxisInterceptLabeler(PointLabeler):
    """Format x/y intercepts independently from other special points."""

    def format_axis_label(
        self,
        x_value: float,
        y_value: float,
        axis: str,
    ) -> str | None:
        if self.style == "No label":
            return None
        if self.style == "Axis value only":
            return format_coordinate(x_value if axis == "x" else y_value)

        coordinates = (
            f"({format_coordinate(x_value)}{self.x_suffix}; "
            f"{format_coordinate(y_value)})"
        )
        if self.style == "Full coordinates":
            return coordinates
        key = _coordinate_key(x_value, y_value)
        letter = self._letters.setdefault(key, _capital_name(len(self._letters)))
        if self.style == "Capital letter only":
            return letter
        return f"{letter}{coordinates}"


def intercepts_enabled(settings: GraphSettings, axis: str) -> bool:
    """Return the independent setting, respecting the legacy override."""

    if settings.show_intercepts is not None:
        return settings.show_intercepts
    return (
        settings.show_x_intercepts
        if axis == "x"
        else settings.show_y_intercepts
    )


def integer_ticks(min_value: float, max_value: float) -> np.ndarray:
    """Return every integer within possibly-decimal visible boundaries."""

    return np.arange(np.ceil(min_value), np.floor(max_value) + 1, 1, dtype=float)


def configure_integer_ticks(ax: Axes, settings: GraphSettings) -> None:
    """Apply exact one-unit integer ticks and labels when requested."""

    if not settings.use_integer_unit_ticks:
        return
    ax.set_xticks(integer_ticks(settings.x_min, settings.x_max))
    ax.set_yticks(integer_ticks(settings.y_min, settings.y_max))
    formatter = FuncFormatter(
        lambda value, _position: format_coordinate(value)
        if abs(value - round(value)) < 1e-9
        else ""
    )
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def graph_label(expression: sp.Expr, function_index: int, style: str) -> str | None:
    """Build a Matplotlib legend label for a plotted function."""

    if style == "No graph label":
        return None

    function_name_index = function_index + 5
    function_name = _capital_name(function_name_index).lower()
    equation = sp.latex(expression)
    if style == "Function equation":
        return f"${function_name}(x) = {equation}$"
    if style == "Function name only":
        return f"${function_name}$"
    return f"$y = {equation}$"


def graph_curve_label(
    expression: sp.Expr,
    function_index: int,
    style: str,
) -> str | None:
    """Build text for a label placed directly beside a curve."""

    if style == "No label":
        return None
    name = _capital_name(function_index + 5).lower()
    if style == "Function name only":
        return f"${name}$"
    if style == "Function notation":
        return f"${name}(x)$"
    return f"${name}(x) = {sp.latex(expression)}$"


def place_graph_curve_label(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    settings: GraphSettings,
    expression: sp.Expr,
    function_index: int = 0,
    *,
    excluded_points: list[tuple[float, float]] | None = None,
    label_override: str | None = None,
) -> None:
    """Place one colour-matched label on a safe visible part of a curve."""

    label = label_override or graph_curve_label(
        expression, function_index, settings.graph_curve_label_style
    )
    if not label:
        return
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    visible = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values >= settings.x_min)
        & (x_values <= settings.x_max)
        & (y_values >= settings.y_min)
        & (y_values <= settings.y_max)
    )
    indices = np.flatnonzero(visible)
    if not indices.size:
        return

    # Prefer the right-hand interior of the curve and stay away from axes and
    # known special points. This also naturally selects just one branch.
    targets = (0.72, 0.62, 0.82, 0.5, 0.35)
    candidates = [indices[min(len(indices) - 1, int(t * (len(indices) - 1)))] for t in targets]
    excluded = excluded_points or []
    x_span = max(settings.x_max - settings.x_min, 1e-9)
    y_span = max(settings.y_max - settings.y_min, 1e-9)

    def score(index: int) -> float:
        x_value, y_value = x_values[index], y_values[index]
        axis_distance = min(abs(x_value) / x_span, abs(y_value) / y_span)
        point_distance = min(
            (
                ((x_value - px) / x_span) ** 2
                + ((y_value - py) / y_span) ** 2
            ) ** 0.5
            for px, py in excluded
        ) if excluded else 1.0
        return axis_distance + point_distance

    index = max(candidates, key=score)
    ax.annotate(
        label,
        (x_values[index], y_values[index]),
        textcoords="offset points",
        xytext=(7, 7),
        color=color,
        fontsize=settings.annotation_font_size,
        bbox=annotation_box(settings),
        annotation_clip=True,
        zorder=9,
    )


def annotation_box(settings: GraphSettings) -> dict[str, object] | None:
    if not settings.annotation_background:
        return None
    return {
        "boxstyle": "round,pad=0.3",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85,
    }


def annotation_arrow(settings: GraphSettings) -> dict[str, str] | None:
    if not settings.annotation_arrows:
        return None
    return {"arrowstyle": "->"}


def annotate_point(
    ax: Axes,
    labeler: PointLabeler,
    settings: GraphSettings,
    x_value: float,
    y_value: float,
    offset: tuple[int, int],
    prefix: str | None = None,
) -> None:
    """Annotate one point according to the shared point-label settings."""

    label = labeler.format_label(x_value, y_value)
    if not label or not labeler.mark_annotated(x_value, y_value):
        return
    if prefix:
        label = f"{prefix} {label}"

    ax.annotate(
        label,
        (x_value, y_value),
        textcoords="offset points",
        xytext=offset,
        fontsize=settings.annotation_font_size,
        bbox=annotation_box(settings),
        arrowprops=annotation_arrow(settings),
        zorder=10,
    )


def annotate_axis_intercept(
    ax: Axes,
    labeler: AxisInterceptLabeler,
    settings: GraphSettings,
    x_value: float,
    y_value: float,
    axis: str,
    offset: tuple[int, int],
) -> None:
    """Annotate an intercept using the dedicated axis-intercept style."""

    label = labeler.format_axis_label(x_value, y_value, axis)
    if not label or not labeler.mark_annotated(x_value, y_value):
        return
    if labeler.style == "Axis value only":
        offset = (0, -16) if axis == "x" else (7, 0)
    ax.annotate(
        label,
        (x_value, y_value),
        textcoords="offset points",
        xytext=offset,
        ha="center" if axis == "x" else "left",
        va="top" if axis == "x" else "center",
        fontsize=settings.annotation_font_size,
        bbox=annotation_box(settings),
        arrowprops=annotation_arrow(settings),
        zorder=10,
    )


def draw_origin_label(ax: Axes, settings: GraphSettings) -> None:
    """Draw one unambiguous zero at the crossing of visible axes."""

    origin_is_visible = (
        settings.show_axes
        and settings.show_origin_label
        and settings.x_min <= 0 <= settings.x_max
        and settings.y_min <= 0 <= settings.y_max
    )
    if not origin_is_visible:
        return
    # An intercept or intersection annotation at the origin already supplies
    # an unambiguous label; do not add a second custom zero on top of it.
    if any(
        getattr(text, "xy", None) is not None
        and np.allclose(text.xy, (0, 0), atol=1e-9)
        and bool(text.get_text())
        for text in ax.texts
    ):
        return

    if settings.show_tick_labels:
        hide_zero = FuncFormatter(
            lambda value, _position: "" if abs(value) < 1e-9 else format_coordinate(value)
        )
        if not isinstance(ax.xaxis.get_major_formatter(), FixedFormatter):
            ax.xaxis.set_major_formatter(hide_zero)
        ax.yaxis.set_major_formatter(hide_zero)

    x_offset = 6 if settings.x_min == 0 else -8
    y_offset = 6 if settings.y_min == 0 else -12
    ax.annotate(
        "0",
        (0, 0),
        textcoords="offset points",
        xytext=(x_offset, y_offset),
        ha="left" if x_offset > 0 else "right",
        va="bottom" if y_offset > 0 else "top",
        fontsize=settings.annotation_font_size,
        zorder=11,
    )


def configure_cartesian_axes(ax: Axes, settings: GraphSettings) -> None:
    """Configure border or school-style Cartesian axes in one shared place."""

    if settings.axis_style not in AXIS_STYLES:
        raise ValueError(f"Axis style must be one of: {', '.join(AXIS_STYLES)}.")

    configure_integer_ticks(ax, settings)
    central = settings.axis_style == "Central Cartesian axes"
    x_origin_visible = settings.x_min <= 0 <= settings.x_max
    y_origin_visible = settings.y_min <= 0 <= settings.y_max

    # Reset positions so the helper is safe to call on reused axes.
    ax.spines["left"].set_position(("outward", 0))
    ax.spines["bottom"].set_position(("outward", 0))
    for spine in ax.spines.values():
        spine.set_visible(settings.show_border)

    if central and settings.show_axes:
        ax.spines["left"].set_position(("data", 0) if x_origin_visible else ("axes", 0))
        ax.spines["bottom"].set_position(
            ("data", 0) if y_origin_visible else ("axes", 0)
        )
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
    elif settings.show_axes:
        if y_origin_visible:
            ax.axhline(0, linewidth=1, color="black", zorder=3)
        if x_origin_visible:
            ax.axvline(0, linewidth=1, color="black", zorder=3)

    axis_content_visible = settings.show_axes or not central
    ticks_visible = settings.show_tick_marks and axis_content_visible
    labels_visible = settings.show_tick_labels and axis_content_visible
    ax.tick_params(
        axis="both",
        which="both",
        bottom=ticks_visible,
        top=ticks_visible and settings.show_border and not central,
        left=ticks_visible,
        right=ticks_visible and settings.show_border and not central,
        labelbottom=labels_visible,
        labeltop=False,
        labelleft=labels_visible,
        labelright=False,
    )

    # Central axes should show at most one normal zero. A custom origin label
    # suppresses both normal zero ticks; otherwise the y-axis zero is hidden.
    if central and labels_visible and x_origin_visible and y_origin_visible:
        hide_zero = FuncFormatter(
            lambda value, _position: (
                "" if abs(value) < 1e-9 else format_coordinate(value)
            )
        )
        ax.yaxis.set_major_formatter(hide_zero)
        if (
            settings.show_origin_label
            and not isinstance(ax.xaxis.get_major_formatter(), FixedFormatter)
        ):
            ax.xaxis.set_major_formatter(hide_zero)

    if not settings.show_axes:
        if central:
            for spine in ax.spines.values():
                spine.set_visible(settings.show_border)
        ax.set_xlabel("")
        ax.set_ylabel("")
        return

    # Put labels beside the axes they name. Standard Matplotlib xlabel/ylabel
    # positions can make a central horizontal axis look like it is labelled
    # "y" and a central vertical axis look like it is labelled "x".
    ax.set_xlabel("")
    ax.set_ylabel("")
    x_span = settings.x_max - settings.x_min
    y_span = settings.y_max - settings.y_min
    arrow_x = settings.x_max - 0.018 * x_span
    arrow_y = settings.y_max - 0.018 * y_span
    horizontal_axis_y = 0 if y_origin_visible else settings.y_min
    vertical_axis_x = 0 if x_origin_visible else settings.x_min

    if central and settings.show_axis_arrows:
        ax.annotate(
            "",
            xy=(arrow_x, horizontal_axis_y),
            xytext=(arrow_x - 0.045 * x_span, horizontal_axis_y),
            arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.2},
            annotation_clip=False,
            zorder=12,
        )
        ax.annotate(
            "",
            xy=(vertical_axis_x, arrow_y),
            xytext=(vertical_axis_x, arrow_y - 0.06 * y_span),
            arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.2},
            annotation_clip=False,
            zorder=12,
        )

    if settings.show_axis_labels:
        horizontal_label = settings.x_axis_label if central else settings.x_label
        vertical_label = settings.y_axis_label if central else settings.y_label
        ax.annotate(
            horizontal_label,
            (arrow_x, horizontal_axis_y),
            textcoords="offset points",
            xytext=(-2, 8),
            ha="right",
            va="bottom",
            fontsize=settings.annotation_font_size + 1,
            annotation_clip=False,
            zorder=13,
        )
        ax.annotate(
            vertical_label,
            (vertical_axis_x, arrow_y),
            textcoords="offset points",
            xytext=(8, -2),
            ha="left",
            va="top",
            fontsize=settings.annotation_font_size + 1,
            annotation_clip=False,
            zorder=13,
        )


def draw_graph_end_arrows(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    settings: GraphSettings,
) -> None:
    """Add arrows just inside both ends of every visible curve segment."""

    if not settings.show_graph_arrows:
        return

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    visible = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values >= settings.x_min)
        & (x_values <= settings.x_max)
        & (y_values >= settings.y_min)
        & (y_values <= settings.y_max)
    )
    visible_indices = np.flatnonzero(visible)
    if visible_indices.size < 4:
        return

    split_at = np.where(np.diff(visible_indices) > 1)[0] + 1
    for segment in np.split(visible_indices, split_at):
        if segment.size < 4:
            continue

        inset = max(1, min(6, segment.size // 5))
        span = max(2, min(18, segment.size // 3))
        start_tip = segment[min(inset, segment.size - 2)]
        start_tail = segment[min(inset + span, segment.size - 1)]
        end_tip = segment[max(0, segment.size - 1 - inset)]
        end_tail = segment[max(0, segment.size - 1 - inset - span)]

        for tail, tip in ((start_tail, start_tip), (end_tail, end_tip)):
            if tail == tip:
                continue
            ax.annotate(
                "",
                xy=(x_values[tip], y_values[tip]),
                xytext=(x_values[tail], y_values[tail]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 2,
                    "mutation_scale": 13,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                annotation_clip=True,
                zorder=6,
            )


def graph_legend_is_enabled(settings: GraphSettings) -> bool:
    return (
        settings.show_legend
        and settings.show_equation
        and settings.graph_label_style != "No graph label"
    )
