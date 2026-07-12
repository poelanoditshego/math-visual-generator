from dataclasses import dataclass


@dataclass
class GraphSettings:
    """
    Stores the customisation settings used when generating a graph.
    """

    # Graph ranges
    x_min: float = -10
    x_max: float = 10
    y_min: float = -10
    y_max: float = 10

    # Graph text
    title: str = ""
    x_label: str = "x"
    y_label: str = "y"

    # General display options
    show_grid: bool = True
    show_axes: bool = True
    show_equation: bool = True
    show_intercepts: bool = True
    show_point_labels: bool = True

    # Linear graph options
    show_gradient: bool = True
    show_gradient_triangle: bool = False

    # Quadratic graph options
    show_turning_point: bool = True
    show_axis_of_symmetry: bool = True
    show_minimum_or_maximum: bool = True

    # Image settings
    figure_width: float = 10
    figure_height: float = 7
    output_name: str = "graph.png"
    image_dpi: int = 300