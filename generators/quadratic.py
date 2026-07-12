from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def create_quadratic_graph(
    equation: str,
    x_min: float = -10,
    x_max: float = 10,
    output_name: str = "quadratic_graph.png",
) -> None:
    
    """
    Generate a quadratic graph and display its important features.

    Parameters:
        equation: A quadratic expression, for example:
                  "x**2 - 4*x + 3"
        x_min: Minimum x-value displayed.
        x_max: Maximum x-value displayed.
        output_name: Name of the generated PNG image.
    """

    x = sp.Symbol("x")

    try:
        expression = sp.sympify(equation)
    except (sp.SympifyError, TypeError) as error:
        raise ValueError(
            "The equation could not be understood. "
            "Use a format such as x**2 - 4*x + 3."
        ) from error

    # Confirm that the expression is quadratic.
    polynomial = sp.Poly(expression, x)

    if polynomial.degree() != 2:
        raise ValueError(
            "This generator only accepts quadratic expressions."
        )

    # Convert the symbolic expression into a NumPy function.
    graph_function = sp.lambdify(x, expression, "numpy")

    # Create x-values and calculate corresponding y-values.
    x_values = np.linspace(x_min, x_max, 1000)
    y_values = graph_function(x_values)

    # Find the important features of the quadratic graph.
    x_intercepts = sp.solve(expression, x)
    y_intercept = expression.subs(x, 0)

    derivative = sp.diff(expression, x)
    turning_x_values = sp.solve(derivative, x)

    # Create the output folder if it does not exist.
    output_folder = Path("generated_graphs")
    output_folder.mkdir(exist_ok=True)

    output_path = output_folder / output_name

    # Create the graph.
    plt.figure(figsize=(10, 7))

    plt.plot(
        x_values,
        y_values,
        linewidth=2,
        label=f"$y = {sp.latex(expression)}$",
    )

    # Draw the coordinate axes.
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    # Add grid lines.
    plt.grid(True, linestyle="--", alpha=0.6)

    # Mark the real x-intercepts.
    for root in x_intercepts:
        if root.is_real:
            root_value = float(root)

            if x_min <= root_value <= x_max:
                plt.scatter(root_value, 0, zorder=5)

                plt.annotate(
                    f"({root_value:g}, 0)",
                    (root_value, 0),
                    textcoords="offset points",
                    xytext=(5, 10),
                )

    # Mark the y-intercept.
    y_intercept_value = float(y_intercept)

    if x_min <= 0 <= x_max:
        plt.scatter(0, y_intercept_value, zorder=5)

        plt.annotate(
            f"(0, {y_intercept_value:g})",
            (0, y_intercept_value),
            textcoords="offset points",
            xytext=(5, 10),
        )

    # Mark the turning point.
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

            plt.annotate(
                (
                    "Turning point "
                    f"({turning_x_value:g}, {turning_y_value:g})"
                ),
                (turning_x_value, turning_y_value),
                textcoords="offset points",
                xytext=(10, -20),
            )

            # Draw the axis of symmetry.
            plt.axvline(
                turning_x_value,
                linestyle=":",
                label=(
                    "Axis of symmetry: "
                    f"x = {turning_x_value:g}"
                ),
            )

    plt.title("Quadratic Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_min, x_max)
    plt.legend()

    plt.tight_layout()

    # Save the graph as an image.
    plt.savefig(output_path, dpi=300)

    # Display the graph.
    plt.show()

    # Close the figure after displaying it.
    plt.close()

    print(f"Graph saved to: {output_path}")