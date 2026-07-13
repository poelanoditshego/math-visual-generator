import streamlit as st

from generators.cosine import create_cosine_graph
from generators.cubic import create_cubic_graph
from generators.exponential import create_exponential_graph
from generators.hyperbola import create_hyperbola_graph
from generators.linear import create_linear_graph
from generators.logarithmic import create_logarithmic_graph
from generators.mixed import create_mixed_graph
from generators.quadratic import create_quadratic_graph
from generators.sine import create_sine_graph
from models.graph_settings import GraphSettings

st.set_page_config(
    page_title="Math Visual Generator",
    page_icon="📈",
    layout="wide",
)

st.title("Math Visual Generator")
st.write(
    "Generate customised linear, quadratic, mixed, exponential, hyperbola, "
    "cubic, logarithmic, sine, and cosine graphs."
)

graph_type = st.selectbox(
    "Choose a graph type",
    [
        "Linear",
        "Quadratic",
        "Mixed",
        "Exponential",
        "Hyperbola",
        "Cubic",
        "Logarithmic",
        "Sine",
        "Cosine",
    ],
)

if graph_type == "Mixed":
    equation_1 = st.text_input(
        "Enter the first equation",
        value="2*x - 4",
    )

    equation_2 = st.text_input(
        "Enter the second equation",
        value="x**2 - 4*x + 3",
    )

else:
    if graph_type == "Linear":
        default_equation = "2*x - 4"
    elif graph_type == "Quadratic":
        default_equation = "x**2 - 4*x + 3"
    elif graph_type == "Exponential":
        default_equation = "2**x"
    elif graph_type == "Hyperbola":
        default_equation = "1/x"
    elif graph_type == "Cubic":
        default_equation = "x**3 - 4*x"
    elif graph_type == "Logarithmic":
        default_equation = "log(x)"
    elif graph_type == "Sine":
        default_equation = "sin(x)"
    else:
        default_equation = "cos(x)"

    equation = st.text_input(
        "Enter the expression",
        value=default_equation,
        key=f"{graph_type.lower()}_equation",
    )

st.subheader("Graph range")

if graph_type in {"Sine", "Cosine"}:
    default_x_min = -360.0
    default_x_max = 360.0
    default_y_min = -5.0
    default_y_max = 5.0
else:
    default_x_min = -10.0
    default_x_max = 10.0
    default_y_min = -10.0
    default_y_max = 10.0

range_col1, range_col2, range_col3, range_col4 = st.columns(4)

with range_col1:
    x_min = st.number_input(
        "Minimum x-value",
        value=default_x_min,
        key=f"{graph_type.lower()}_x_min",
    )

with range_col2:
    x_max = st.number_input(
        "Maximum x-value",
        value=default_x_max,
        key=f"{graph_type.lower()}_x_max",
    )

with range_col3:
    y_min = st.number_input(
        "Minimum y-value",
        value=default_y_min,
        key=f"{graph_type.lower()}_y_min",
    )

with range_col4:
    y_max = st.number_input(
        "Maximum y-value",
        value=default_y_max,
        key=f"{graph_type.lower()}_y_max",
    )

st.subheader("Graph labels")

title = st.text_input(
    "Graph title",
    value=f"{graph_type} Function",
    key=f"{graph_type.lower()}_title",
)

label_col1, label_col2 = st.columns(2)

with label_col1:
    x_label = st.text_input(
        "x-axis label",
        value="x",
    )

with label_col2:
    y_label = st.text_input(
        "y-axis label",
        value="y",
    )

graph_label_style = st.selectbox(
    "Graph label style",
    [
        "Full equation",
        "Function equation",
        "Function name only",
        "No graph label",
    ],
)

st.subheader("Display options")

option_col1, option_col2, option_col3 = st.columns(3)

with option_col1:
    show_grid = st.checkbox(
        "Show grid",
        value=True,
    )

    show_axes = st.checkbox(
        "Show axes",
        value=True,
    )

    show_border = st.checkbox(
        "Show graph boundary",
        value=True,
    )

    show_title = st.checkbox(
        "Show graph title",
        value=True,
    )

with option_col2:
    show_equation = st.checkbox(
        "Show equation",
        value=True,
    )

    show_intercepts = st.checkbox(
        "Show intercepts",
        value=True,
    )

    show_tick_marks = st.checkbox(
        "Show tick marks",
        value=True,
    )

    show_legend = st.checkbox(
        "Show legend",
        value=True,
    )

with option_col3:
    show_tick_labels = st.checkbox(
        "Show axis numbers",
        value=True,
    )

    show_intersection_points = st.checkbox(
        "Show intersection points",
        value=True,
    )

st.subheader("Point labels")

point_label_style = st.selectbox(
    "Point label style",
    [
        "Coordinates only",
        "Capital letter and coordinates",
        "Capital letter only",
        "No label",
    ],
)

point_label_col1, point_label_col2 = st.columns(2)

with point_label_col1:
    show_point_labels = st.checkbox(
        "Show point labels",
        value=True,
    )

with point_label_col2:
    annotation_background = st.checkbox(
        "Show label backgrounds",
        value=True,
    )

st.subheader("Axes and origin")

show_origin_label = st.checkbox(
    "Show 0 at the origin",
    value=True,
)

st.subheader("Graph arrows")

show_graph_arrows = st.checkbox(
    "Show arrows at graph ends",
    value=True,
)

st.subheader("Label positions")

position_col1, position_col2 = st.columns(2)

with position_col1:
    x_intercept_horizontal = st.number_input(
        "X-intercept horizontal offset",
        value=8,
    )

    x_intercept_vertical = st.number_input(
        "X-intercept vertical offset",
        value=12,
    )

with position_col2:
    y_intercept_horizontal = st.number_input(
        "Y-intercept horizontal offset",
        value=-50,
    )

    y_intercept_vertical = st.number_input(
        "Y-intercept vertical offset",
        value=10,
    )

st.subheader("Intersection label position")

intersection_col1, intersection_col2 = st.columns(2)

with intersection_col1:
    intersection_label_horizontal = st.number_input(
        "Intersection label horizontal offset",
        value=8,
    )

with intersection_col2:
    intersection_label_vertical = st.number_input(
        "Intersection label vertical offset",
        value=12,
    )

show_gradient = True
show_gradient_triangle = False
show_turning_point = True
show_axis_of_symmetry = True
show_horizontal_asymptote = True
horizontal_asymptote_label = True
show_vertical_asymptote = True
show_asymptote_labels = True
show_hyperbola_centre = True
show_stationary_points = True
show_stationary_point_labels = True
show_stationary_point_type = False
show_inflection_point = True
show_inflection_point_label = True
trig_angle_mode = "Degrees"
show_pi_tick_labels = False
show_degree_symbols = True
show_midline = True
show_midline_label = True
show_maximum_points = True
show_minimum_points = True
show_extreme_point_labels = True
show_sine_key_points = False
show_standard_trig_points = False
show_cosine_key_points = False
turning_point_horizontal = 10
turning_point_vertical = -25

if graph_type == "Linear":
    st.subheader("Linear graph options")

    show_gradient = st.checkbox(
        "Show gradient",
        value=True,
    )

    show_gradient_triangle = st.checkbox(
        "Show gradient triangle",
        value=True,
    )

elif graph_type in {"Quadratic", "Mixed"}:
    st.subheader("Quadratic graph options")

    show_turning_point = st.checkbox(
        "Show turning point",
        value=True,
    )

    show_axis_of_symmetry = st.checkbox(
        "Show axis of symmetry",
        value=True,
    )

    turning_col1, turning_col2 = st.columns(2)

    with turning_col1:
        turning_point_horizontal = st.number_input(
            "Turning-point horizontal offset",
            value=10,
        )

    with turning_col2:
        turning_point_vertical = st.number_input(
            "Turning-point vertical offset",
            value=-25,
        )

elif graph_type == "Exponential":
    st.subheader("Exponential graph options")

    show_horizontal_asymptote = st.checkbox(
        "Show horizontal asymptote",
        value=True,
    )

    horizontal_asymptote_label = st.checkbox(
        "Show asymptote label",
        value=True,
    )

elif graph_type == "Hyperbola":
    st.subheader("Hyperbola graph options")

    show_vertical_asymptote = st.checkbox(
        "Show vertical asymptote",
        value=True,
    )

    show_horizontal_asymptote = st.checkbox(
        "Show horizontal asymptote",
        value=True,
    )

    show_asymptote_labels = st.checkbox(
        "Show asymptote labels",
        value=True,
    )

    show_hyperbola_centre = st.checkbox(
        "Show hyperbola centre",
        value=True,
    )

elif graph_type == "Cubic":
    st.subheader("Cubic graph options")

    show_stationary_points = st.checkbox(
        "Show stationary points",
        value=True,
    )

    show_stationary_point_labels = st.checkbox(
        "Show stationary-point labels",
        value=True,
    )

    show_stationary_point_type = st.checkbox(
        "Show stationary-point type",
        value=False,
    )

    show_inflection_point = st.checkbox(
        "Show point of inflection",
        value=True,
    )

    show_inflection_point_label = st.checkbox(
        "Show inflection-point label",
        value=True,
    )

elif graph_type == "Logarithmic":
    st.subheader("Logarithmic graph options")

    show_vertical_asymptote = st.checkbox(
        "Show vertical asymptote",
        value=True,
    )

    show_asymptote_labels = st.checkbox(
        "Show asymptote labels",
        value=True,
    )

elif graph_type in {"Sine", "Cosine"}:
    st.subheader(f"{graph_type} graph options")

    trig_angle_mode = st.selectbox(
        "Angle mode",
        ["Degrees", "Radians"],
    )

    if trig_angle_mode == "Degrees":
        show_degree_symbols = st.checkbox(
            "Show degree symbols",
            value=True,
        )

    elif graph_type == "Cosine":
        show_pi_tick_labels = st.checkbox(
            "Show x-axis labels as multiples of pi",
            value=True,
        )

    show_midline = st.checkbox(
        "Show midline",
        value=True,
    )

    show_midline_label = st.checkbox(
        "Show midline label",
        value=True,
    )

    show_maximum_points = st.checkbox(
        "Show maximum points",
        value=True,
    )

    show_minimum_points = st.checkbox(
        "Show minimum points",
        value=True,
    )

    show_extreme_point_labels = st.checkbox(
        "Show maximum and minimum labels",
        value=True,
    )

    show_standard_trig_points = st.checkbox(
        "Show standard trigonometric points",
        value=False,
    )
    show_sine_key_points = show_standard_trig_points
    show_cosine_key_points = show_standard_trig_points


st.subheader("Additional coordinates")

additional_x_input = st.text_input(
    "Enter additional x-values separated by commas",
    value="",
    placeholder="-2, 1, 4",
)

show_additional_point_labels = st.checkbox(
    "Show additional coordinate labels",
    value=True,
)

additional_col1, additional_col2 = st.columns(2)

with additional_col1:
    additional_label_x_offset = st.number_input(
        "Additional-label horizontal offset",
        value=8,
    )

with additional_col2:
    additional_label_y_offset = st.number_input(
        "Additional-label vertical offset",
        value=12,
    )

if graph_type == "Linear":
    default_output_name = "linear_graph.png"

elif graph_type == "Quadratic":
    default_output_name = "quadratic_graph.png"

elif graph_type == "Mixed":
    default_output_name = "mixed_graph.png"

elif graph_type == "Exponential":
    default_output_name = "exponential_graph.png"

elif graph_type == "Hyperbola":
    default_output_name = "hyperbola_graph.png"

elif graph_type == "Cubic":
    default_output_name = "cubic_graph.png"

elif graph_type == "Logarithmic":
    default_output_name = "logarithmic_graph.png"

elif graph_type == "Sine":
    default_output_name = "sine_graph.png"

else:
    default_output_name = "cosine_graph.png"

output_name = st.text_input(
    "Output filename",
    value=default_output_name,
    key=f"{graph_type.lower()}_output_name",
)

if st.button("Generate Graph", type="primary"):
    if x_min >= x_max:
        st.error(
            "The minimum x-value must be smaller than the maximum."
        )

    elif y_min >= y_max:
        st.error(
            "The minimum y-value must be smaller than the maximum."
        )

    else:

        additional_x_values = []

        if additional_x_input.strip():
            try:
                additional_x_values = [
                    float(value.strip())
                    for value in additional_x_input.split(",")
                ]
            except ValueError:
                st.error(
                    "Additional x-values must be numbers separated by commas."
                )
                st.stop()


        settings = GraphSettings(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            title=title,
            x_label=x_label,
            y_label=y_label,

            show_grid=show_grid,
            show_axes=show_axes,
            show_equation=show_equation,

            show_title=show_title,
            show_legend=show_legend,

            show_intercepts=show_intercepts,
            show_intersection_points=show_intersection_points,
            show_point_labels=show_point_labels,
            point_label_style=point_label_style,
            graph_label_style=graph_label_style,
            show_origin_label=show_origin_label,
            show_graph_arrows=show_graph_arrows,

            show_border=show_border,
            show_tick_marks=show_tick_marks,
            show_tick_labels=show_tick_labels,

            show_gradient=show_gradient,
            show_gradient_triangle=show_gradient_triangle,
            show_turning_point=show_turning_point,
            show_axis_of_symmetry=show_axis_of_symmetry,
            show_horizontal_asymptote=show_horizontal_asymptote,
            horizontal_asymptote_label=horizontal_asymptote_label,
            show_vertical_asymptote=show_vertical_asymptote,
            show_asymptote_labels=show_asymptote_labels,
            show_hyperbola_centre=show_hyperbola_centre,
            show_stationary_points=show_stationary_points,
            show_stationary_point_labels=show_stationary_point_labels,
            show_stationary_point_type=show_stationary_point_type,
            show_inflection_point=show_inflection_point,
            show_inflection_point_label=show_inflection_point_label,
            trig_angle_mode=trig_angle_mode,
            show_pi_tick_labels=show_pi_tick_labels,
            show_degree_symbols=show_degree_symbols,
            show_midline=show_midline,
            show_midline_label=show_midline_label,
            show_maximum_points=show_maximum_points,
            show_minimum_points=show_minimum_points,
            show_extreme_point_labels=show_extreme_point_labels,
            show_sine_key_points=show_sine_key_points,
            show_standard_trig_points=show_standard_trig_points,
            show_cosine_key_points=show_cosine_key_points,
            x_intercept_label_offset=(
                int(x_intercept_horizontal),
                int(x_intercept_vertical),
            ),
            y_intercept_label_offset=(
                int(y_intercept_horizontal),
                int(y_intercept_vertical),
            ),
            turning_point_label_offset=(
                int(turning_point_horizontal),
                int(turning_point_vertical),
            ),

            intersection_label_offset=(
                int(intersection_label_horizontal),
                int(intersection_label_vertical),
            ),

            additional_x_values=additional_x_values,
            show_additional_point_labels=show_additional_point_labels,
            additional_point_label_offset=(
                int(additional_label_x_offset),
                int(additional_label_y_offset),
            ),

            annotation_background=annotation_background,
            output_name=output_name,
        )

        try:
            if graph_type == "Linear":
                create_linear_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Quadratic":
                create_quadratic_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Exponential":
                create_exponential_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Hyperbola":
                create_hyperbola_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Cubic":
                create_cubic_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Logarithmic":
                create_logarithmic_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Sine":
                create_sine_graph(
                    equation=equation,
                    settings=settings,
                )

            elif graph_type == "Cosine":
                create_cosine_graph(
                    equation=equation,
                    settings=settings,
                )

            else:
                create_mixed_graph(
                    equations=[
                        equation_1,
                        equation_2,
                    ],
                    settings=settings,
                )

            graph_path = f"generated_graphs/{output_name}"

            st.success("Graph generated successfully.")
            st.image(
                graph_path,
                caption=title,
                width="stretch",
            )

            with open(graph_path, "rb") as graph_file:
                st.download_button(
                    label="Download graph",
                    data=graph_file,
                    file_name=output_name,
                    mime="image/png",
                )

        except ValueError as error:
            st.error(str(error))
