from __future__ import annotations

import re
from string import ascii_uppercase

import numpy as np
import sympy as sp
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from models.graph_settings import GraphSettings


POINT_LABEL_STYLES = (
    "Coordinates only",
    "Capital letter and coordinates",
    "Capital letter only",
    "No label",
)

GRAPH_LABEL_STYLES = (
    "Full equation",
    "Function equation",
    "Function name only",
    "No graph label",
)

_SAFE_ARITHMETIC_EXPRESSION = re.compile(r"[0-9x+\-*/().\s]+")


def parse_arithmetic_expression(
    equation: str,
    graph_name: str,
    example: str,
) -> tuple[sp.Symbol, sp.Expr]:
    """Safely parse a one-variable arithmetic expression for a generator."""

    if not isinstance(equation, str) or not equation.strip():
        raise ValueError(f"Enter a {graph_name.lower()} expression such as {example}.")

    identifiers = set(re.findall(r"[A-Za-z_]\w*", equation))
    unsupported_identifiers = identifiers - {"x"}
    if unsupported_identifiers:
        names = ", ".join(sorted(unsupported_identifiers))
        raise ValueError(
            f"{graph_name} expressions may only use the variable x; "
            f"unsupported name: {names}."
        )
    if (
        len(equation) > 500
        or _SAFE_ARITHMETIC_EXPRESSION.fullmatch(equation) is None
    ):
        raise ValueError(
            "The expression contains unsupported characters or functions. "
            "Use numbers, x, parentheses, and arithmetic operators only."
        )

    x = sp.Symbol("x", real=True)
    try:
        expression = sp.sympify(equation, locals={"x": x})
    except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"The equation could not be understood. Use a format such as {example}."
        ) from error

    if expression.free_symbols - {x}:
        raise ValueError(f"{graph_name} expressions may only contain the variable x.")
    if expression.atoms(sp.Function):
        raise ValueError("Named functions are not supported by this graph generator.")

    return x, expression


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

    def __init__(self, style: str) -> None:
        self.style = style
        self._letters: dict[tuple[float, float], str] = {}
        self._annotated: set[tuple[float, float]] = set()

    def format_label(self, x_value: float, y_value: float) -> str | None:
        if self.style == "No label":
            return None

        coordinates = (
            f"({format_coordinate(x_value)}, {format_coordinate(y_value)})"
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
) -> None:
    """Annotate one point according to the shared point-label settings."""

    label = labeler.format_label(x_value, y_value)
    if not label or not labeler.mark_annotated(x_value, y_value):
        return

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

    if settings.show_tick_labels:
        hide_zero = FuncFormatter(
            lambda value, _position: "" if abs(value) < 1e-9 else format_coordinate(value)
        )
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
