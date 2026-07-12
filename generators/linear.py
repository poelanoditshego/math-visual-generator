from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def create_linear_graph(
    equation: str,
    x_min: float = -10,
    x_max: float = 10,
    output_name: str = "linear_graph.png",
) -> None:
    """
    Generate a linear graph and mark its intercepts.
    """

    x = sp.Symbol("x")

    try:
        expression = sp.sympify(equation)
    except (sp.SympifyError, TypeError) as error:
        raise ValueError(
            "The equation could not be understood. "
            "Use a format such as 2*x - 4."
        ) from error

    polynomial = sp.Poly(expression, x)

    if polynomial.degree() != 1:
        raise ValueError(
            "This generator only accepts linear expressions."
        )

    graph_function = sp.lambdify(x, expression, "numpy")

    x_values = np.linspace(x_min, x_max, 1000)
    y_values = graph_function(x_values)

    x_intercepts = sp.solve(expression, x)
    y_intercept = expression.subs(x, 0)

    output_folder = Path("generated_graphs")
    output_folder.mkdir(exist_ok=True)

    output_path = output_folder / output_name

    plt.figure(figsize=(10, 7))

    plt.plot(
        x_values,
        y_values,
        linewidth=2,
        label=f"$y = {sp.latex(expression)}$",
    )

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.grid(True, linestyle="--", alpha=0.6)

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

    y_intercept_value = float(y_intercept)

    if x_min <= 0 <= x_max:
        plt.scatter(0, y_intercept_value, zorder=5)

        plt.annotate(
            f"(0, {y_intercept_value:g})",
            (0, y_intercept_value),
            textcoords="offset points",
            xytext=(5, 10),
        )

    plt.title("Linear Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_min, x_max)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Graph saved to: {output_path}")