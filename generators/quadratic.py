from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from models.graph_settings import GraphSettings


def create_quadratic_graph(
    equation: str,
    settings: GraphSettings,
) -> None:
    x = sp.Symbol("x")

    try:
        expression = sp.sympify(equation)
    except (sp.SympifyError, TypeError) as error:
        raise ValueError(
            "The equation could not be understood. "
            "Use a format such as x**2 - 4*x + 3."
        ) from error

    polynomial = sp.Poly(expression, x)

    if polynomial.degree() != 2:
        raise ValueError(
            "This generator only accepts quadratic expressions."
        )

    graph_function = sp.lambdify(x, expression, "numpy")

    x_values = np.linspace(
        settings.x_min,
        settings.x_max,
        1000,
    )

    y_values = graph_function(x_values)

    x_intercepts = sp.solve(expression, x)
    y_intercept = expression.subs(x, 0)

    derivative = sp.diff(expression, x)
    turning_x_values = sp.solve(derivative, x)

    output_folder = Path("generated_graphs")
    output_folder.mkdir(exist_ok=True)

    output_path = output_folder / settings.output_name

    plt.figure(
        figsize=(
            settings.figure_width,
            settings.figure_height,
        )
    )

    ax = plt.gca()

    # Show or hide the black boundary around the graph.
    for spine in ax.spines.values():
        spine.set_visible(settings.show_border)

    # Show or hide tick marks and number labels.
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

    equation_label = None

    if settings.show_equation:
        equation_label = f"$y = {sp.latex(expression)}$"

    plt.plot(
        x_values,
        y_values,
        linewidth=2,
        label=equation_label,
    )

    if settings.show_axes:
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)

    if settings.show_grid:
        plt.grid(
            True,
            linestyle="--",
            alpha=0.6,
        )

    if settings.show_intercepts:
        for root in x_intercepts:
            if root.is_real:
                root_value = float(root)

                if settings.x_min <= root_value <= settings.x_max:
                    plt.scatter(
                        root_value,
                        0,
                        zorder=5,
                    )

                    if settings.show_point_labels:
                        annotation_box = None
                        annotation_arrow = None

                        if settings.annotation_background:
                            annotation_box = {
                                "boxstyle": "round,pad=0.3",
                                "facecolor": "white",
                                "alpha": 0.85,
                            }

                        if settings.annotation_arrows:
                            annotation_arrow = {
                                "arrowstyle": "->",
                            }



                        plt.annotate(
                            f"({root_value:g}, 0)",
                            (root_value, 0),
                            textcoords="offset points",
                            xytext=settings.x_intercept_label_offset,
                            fontsize=settings.annotation_font_size,
                            bbox=annotation_box,
                            arrowprops=annotation_arrow,
                            zorder=10,
                        )

        y_intercept_value = float(y_intercept)

        if settings.x_min <= 0 <= settings.x_max:
            plt.scatter(
                0,
                y_intercept_value,
                zorder=5,
            )

            if settings.show_point_labels:
                plt.annotate(
                    f"(0, {y_intercept_value:g})",
                    (0, y_intercept_value),
                    textcoords="offset points",
                    xytext=settings.y_intercept_label_offset,
                    fontsize=settings.annotation_font_size,
                    bbox=annotation_box,
                    arrowprops=annotation_arrow,
                    zorder=10,
                )

    if settings.show_turning_point:
        for turning_x in turning_x_values:
            if turning_x.is_real:
                turning_x_value = float(turning_x)
                turning_y_value = float(
                    expression.subs(x, turning_x)
                )

                plt.scatter(
                    turning_x_value,
                    turning_y_value,
                    zorder=5,
                )

                if settings.show_point_labels:
                    plt.annotate(
                        (
                            "Turning point "
                            f"({turning_x_value:g}, {turning_y_value:g})"
                        ),
                        (turning_x_value, turning_y_value),
                        textcoords="offset points",
                        xytext=settings.turning_point_label_offset,
                        fontsize=settings.annotation_font_size,
                        bbox=annotation_box,
                        arrowprops=annotation_arrow,
                        zorder=10,
                    )

                if settings.show_axis_of_symmetry:
                    plt.axvline(
                        turning_x_value,
                        linestyle=":",
                        label=(
                            "Axis of symmetry: "
                            f"x = {turning_x_value:g}"
                        ),
                    )

    # Draw additional points selected by the user.
    for additional_x in settings.additional_x_values:
        additional_y = float(
            expression.subs(x, additional_x)
        )

        point_is_visible = (
            settings.x_min <= additional_x <= settings.x_max
            and settings.y_min <= additional_y <= settings.y_max
        )

        if point_is_visible:
            plt.scatter(
                additional_x,
                additional_y,
                zorder=6,
            )

            if settings.show_additional_point_labels:
                plt.annotate(
                    f"({additional_x:g}, {additional_y:g})",
                    (additional_x, additional_y),
                    textcoords="offset points",
                    xytext=settings.additional_point_label_offset,
                    fontsize=settings.annotation_font_size,
                    bbox=annotation_box,
                    arrowprops=annotation_arrow,
                    zorder=10,
                )


    if settings.show_title:
        plt.title(
            settings.title or "Quadratic Function"
        )

    plt.xlabel(settings.x_label)
    plt.ylabel(settings.y_label)

    plt.xlim(
        settings.x_min,
        settings.x_max,
    )

    plt.ylim(
        settings.y_min,
        settings.y_max,
    )

    if settings.show_legend and (
    settings.show_equation
    or settings.show_axis_of_symmetry
    ):
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=settings.image_dpi,
    )

    plt.show()
    plt.close()

    print(f"Graph saved to: {output_path}")