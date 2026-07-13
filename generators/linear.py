from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from models.graph_settings import GraphSettings


def create_linear_graph(
    equation: str,
    settings: GraphSettings,
) -> None:
    """
    Generate a linear graph using the supplied customisation settings.
    """

    x = sp.Symbol("x")

    try:
        expression = sp.sympify(equation)
        polynomial = sp.Poly(expression, x)
    except (sp.SympifyError, sp.PolynomialError, TypeError) as error:
        raise ValueError(
            "The equation could not be understood. "
            "Use a format such as 2*x - 4."
        ) from error

    # Confirm that the expression is linear.
    if polynomial.degree() != 1:
        raise ValueError(
            "This generator only accepts linear expressions."
        )

    # Extract the gradient and y-intercept from y = mx + c.
    gradient = polynomial.coeff_monomial(x)
    y_intercept = expression.subs(x, 0)

    # Convert the symbolic expression into a NumPy function.
    graph_function = sp.lambdify(x, expression, "numpy")

    x_values = np.linspace(
        settings.x_min,
        settings.x_max,
        1000,
    )

    y_values = graph_function(x_values)

    # Find the x-intercept by solving y = 0.
    x_intercepts = sp.solve(expression, x)

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

    # Mark the x-intercept and y-intercept.
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

        if (
            settings.x_min <= 0 <= settings.x_max
            and settings.y_min <= y_intercept_value <= settings.y_max
        ):
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

    # Add the gradient to the graph legend.
    if settings.show_gradient:
        gradient_label = f"Gradient: m = {float(gradient):g}"

        plt.plot(
            [],
            [],
            linestyle="",
            label=gradient_label,
        )

    # Draw a simple gradient triangle.
    if settings.show_gradient_triangle:
        gradient_value = float(gradient)
        start_x = 0.0
        start_y = float(y_intercept)

        run = 1.0
        rise = gradient_value * run

        end_x = start_x + run
        end_y = start_y + rise

        triangle_is_visible = (
            settings.x_min <= start_x <= settings.x_max
            and settings.x_min <= end_x <= settings.x_max
            and settings.y_min <= start_y <= settings.y_max
            and settings.y_min <= end_y <= settings.y_max
        )

        if triangle_is_visible:
            # Horizontal part of the triangle: run.
            plt.plot(
                [start_x, end_x],
                [start_y, start_y],
                linestyle="--",
            )

            # Vertical part of the triangle: rise.
            plt.plot(
                [end_x, end_x],
                [start_y, end_y],
                linestyle="--",
            )

            if settings.show_point_labels:
                plt.annotate(
                    f"Run = {run:g}",
                    (
                        (start_x + end_x) / 2,
                        start_y,
                    ),
                    textcoords="offset points",
                    xytext=(0, -18),
                    ha="center",
                )

                plt.annotate(
                    f"Rise = {rise:g}",
                    (
                        end_x,
                        (start_y + end_y) / 2,
                    ),
                    textcoords="offset points",
                    xytext=(8, 0),
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
            settings.title or "Linear Function"
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
    settings.show_equation or settings.show_gradient
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