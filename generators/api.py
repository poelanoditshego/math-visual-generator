from pathlib import Path
from typing import Callable

from constants import SUPPORTED_GRAPH_TYPES
from generators.circle import create_circle_graph
from generators.cosine import create_cosine_graph
from generators.cubic import create_cubic_graph
from generators.exponential import create_exponential_graph
from generators.hyperbola import create_hyperbola_graph
from generators.linear import create_linear_graph
from generators.logarithmic import create_logarithmic_graph
from generators.mixed import create_mixed_graph
from generators.quadratic import create_quadratic_graph
from generators.sine import create_sine_graph
from generators.tangent import create_tangent_graph
from models.graph_artifact import GraphArtifact
from models.graph_request import GraphRequest, validate_output_name
from models.graph_settings import GraphSettings


_SINGLE_EQUATION_GENERATORS: dict[str, Callable[[str, GraphSettings], None]] = {
    "Linear": create_linear_graph,
    "Quadratic": create_quadratic_graph,
    "Exponential": create_exponential_graph,
    "Hyperbola": create_hyperbola_graph,
    "Cubic": create_cubic_graph,
    "Logarithmic": create_logarithmic_graph,
    "Sine": create_sine_graph,
    "Cosine": create_cosine_graph,
    "Tangent": create_tangent_graph,
    "Circle": create_circle_graph,
}


def graph_request_to_settings(
    request: GraphRequest,
    *,
    output_directory: Path | str | None = None,
) -> GraphSettings:
    """Convert the public request controls into internal graph settings."""

    display = request.display
    graph_range = request.graph_range
    return GraphSettings(
        x_min=graph_range.x_min,
        x_max=graph_range.x_max,
        y_min=graph_range.y_min,
        y_max=graph_range.y_max,
        show_grid=display.show_grid,
        show_axes=display.show_axes,
        show_equation=display.show_equation,
        show_title=display.show_title,
        show_legend=display.show_legend,
        show_border=display.show_border,
        show_tick_marks=display.show_tick_marks,
        show_tick_labels=display.show_tick_labels,
        use_integer_unit_ticks=display.use_integer_unit_ticks,
        show_x_intercepts=display.show_x_intercepts,
        show_y_intercepts=display.show_y_intercepts,
        show_intersection_points=display.show_intersection_points,
        show_point_labels=display.show_point_labels,
        point_label_style=display.point_label_style,
        axis_intercept_label_style=display.axis_intercept_label_style,
        graph_curve_label_style=display.graph_curve_label_style,
        show_origin_label=display.show_origin_label,
        show_graph_arrows=display.show_graph_arrows,
        show_axis_arrows=display.show_axis_arrows,
        show_axis_labels=display.show_axis_labels,
        additional_x_values=display.additional_x_values,
        additional_point_labels=display.additional_point_labels,
        additional_point_function_indices=display.additional_point_function_indices,
        show_additional_point_labels=display.show_additional_point_labels,
        show_gradient=display.show_gradient,
        show_gradient_triangle=display.show_gradient_triangle,
        show_turning_point=display.show_turning_point,
        show_axis_of_symmetry=display.show_axis_of_symmetry,
        show_vertical_asymptote=display.show_vertical_asymptote,
        show_horizontal_asymptote=display.show_horizontal_asymptote,
        horizontal_asymptote_label=display.show_asymptote_labels,
        show_asymptote_labels=display.show_asymptote_labels,
        show_hyperbola_centre=display.show_hyperbola_centre,
        show_stationary_points=display.show_stationary_points,
        show_inflection_point=display.show_inflection_point,
        trig_angle_mode=display.trig_angle_mode,
        show_midline=display.show_midline,
        show_maximum_points=display.show_maximum_points,
        show_minimum_points=display.show_minimum_points,
        show_circle_centre=display.show_circle_centre,
        show_radius=display.show_radius,
        show_radius_label=display.show_radius_label,
        show_diameter=display.show_diameter,
        show_diameter_label=display.show_diameter_label,
        output_name=request.output_name,
        output_directory=Path(output_directory) if output_directory else Path("generated_graphs"),
    )


def generate_graph(
    graph_type: str,
    settings: GraphSettings,
    equation: str | None = None,
    equations: list[str] | None = None,
) -> GraphArtifact:
    """Generate one supported graph and return its created image artifact."""

    if graph_type not in SUPPORTED_GRAPH_TYPES:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    validate_output_name(settings.output_name)

    if graph_type == "Mixed":
        if equation is not None:
            raise ValueError("Mixed graph generation must use equations only.")
        if not isinstance(equations, list) or len(equations) != 2:
            raise ValueError("Mixed graph generation requires exactly two equations.")
        if any(not isinstance(item, str) or not item.strip() for item in equations):
            raise ValueError("Mixed graph generation requires two non-empty equations.")
        create_mixed_graph(equations=equations, settings=settings)
    else:
        if not isinstance(equation, str) or not equation.strip():
            raise ValueError(f"An equation is required for {graph_type} graph generation.")
        _SINGLE_EQUATION_GENERATORS[graph_type](
            equation=equation,
            settings=settings,
        )

    image_path = Path(settings.output_directory) / settings.output_name
    if not image_path.is_file():
        raise ValueError("Graph generation completed but no output image was created.")

    return GraphArtifact(
        image_path=str(image_path),
        graph_type=graph_type,
    )


def generate_graph_from_request(
    request: GraphRequest,
    *,
    output_directory: Path | str | None = None,
) -> GraphArtifact:
    """Validate a structured request, convert it, and generate its graph."""

    request.validate()
    settings = graph_request_to_settings(request, output_directory=output_directory)
    return generate_graph(
        graph_type=request.graph_type,
        equation=request.equation,
        equations=request.equations,
        settings=settings,
    )
