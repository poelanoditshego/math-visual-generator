from dataclasses import dataclass, field


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

      # Annotation settings
    x_intercept_label_offset: tuple[int, int] = (8, 12)
    y_intercept_label_offset: tuple[int, int] = (8, 12)
    turning_point_label_offset: tuple[int, int] = (10, -25)

    annotation_font_size: int = 10
    annotation_background: bool = True
    annotation_arrows: bool = False

    # Additional coordinates
    additional_x_values: list[float] = field(default_factory=list)
    show_additional_point_labels: bool = True
    additional_point_label_offset: tuple[int, int] = (8, 12)

    # Axis boundary and numbering
    show_border: bool = True
    show_tick_marks: bool = True
    show_tick_labels: bool = True

    show_title: bool = True
    show_legend: bool = True