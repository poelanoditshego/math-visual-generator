import streamlit as st

from generators.linear import create_linear_graph
from generators.quadratic import create_quadratic_graph
from models.graph_settings import GraphSettings


st.set_page_config(
    page_title="Math Visual Generator",
    page_icon="📈",
    layout="wide",
)

st.title("Math Visual Generator")
st.write("Generate customised linear and quadratic graphs.")

graph_type = st.selectbox(
    "Choose a graph type",
    ["Linear", "Quadratic"],
)

if graph_type == "Linear":
    default_equation = "2*x - 4"
else:
    default_equation = "x**2 - 4*x + 3"

equation = st.text_input(
    "Enter the expression",
    value=default_equation,
)

st.subheader("Graph range")

range_col1, range_col2, range_col3, range_col4 = st.columns(4)

with range_col1:
    x_min = st.number_input(
        "Minimum x-value",
        value=-10.0,
    )

with range_col2:
    x_max = st.number_input(
        "Maximum x-value",
        value=10.0,
    )

with range_col3:
    y_min = st.number_input(
        "Minimum y-value",
        value=-10.0,
    )

with range_col4:
    y_max = st.number_input(
        "Maximum y-value",
        value=10.0,
    )

st.subheader("Graph labels")

title = st.text_input(
    "Graph title",
    value=f"{graph_type} Function",
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
    show_point_labels = st.checkbox(
        "Show point labels",
        value=True,
    )

    annotation_background = st.checkbox(
        "Show label backgrounds",
        value=True,
    )

    show_tick_labels = st.checkbox(
        "Show axis numbers",
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

show_gradient = True
show_gradient_triangle = False
show_turning_point = True
show_axis_of_symmetry = True
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

else:
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

output_name = st.text_input(
    "Output filename",
    value=(
        "linear_graph.png"
        if graph_type == "Linear"
        else "quadratic_graph.png"
    ),
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
            show_point_labels=show_point_labels,

            show_border=show_border,
            show_tick_marks=show_tick_marks,
            show_tick_labels=show_tick_labels,

            show_gradient=show_gradient,
            show_gradient_triangle=show_gradient_triangle,
            show_turning_point=show_turning_point,
            show_axis_of_symmetry=show_axis_of_symmetry,
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
            else:
                create_quadratic_graph(
                    equation=equation,
                    settings=settings,
                )

            graph_path = f"generated_graphs/{output_name}"

            st.success("Graph generated successfully.")
            st.image(
                graph_path,
                caption=title,
                use_container_width=True,
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