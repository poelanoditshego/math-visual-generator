from generators.linear import create_linear_graph
from generators.quadratic import create_quadratic_graph
from models.graph_settings import GraphSettings


def main() -> None:
    print("Math Visual Generator")
    print("---------------------")
    print("1. Linear graph")
    print("2. Quadratic graph")

    graph_choice = input("Choose a graph type: ").strip()

    if graph_choice not in {"1", "2"}:
        print("Invalid choice. Enter 1 or 2.")
        return

    equation = input("Enter the expression: ").strip()

    settings = GraphSettings(
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,

        show_grid=True,
        show_axes=True,
        show_equation=True,
        show_intercepts=True,
        show_point_labels=True,

        show_border=False,
        show_tick_marks=True,
        show_tick_labels=True,

        annotation_font_size=10,
        annotation_background=True,
        annotation_arrows=False,
        
        x_intercept_label_offset=(8, 12),
        y_intercept_label_offset=(-50, 10),
    )

    try:
        if graph_choice == "1":
            settings.title = "Linear Function"
            settings.output_name = "custom_linear.png"
            settings.show_gradient = True
            settings.show_gradient_triangle = True

            create_linear_graph(
                equation=equation,
                settings=settings,
            )

        elif graph_choice == "2":
            settings.title = "Quadratic Function"
            settings.output_name = "custom_quadratic.png"
            settings.show_turning_point = True
            settings.show_axis_of_symmetry = True
            settings.turning_point_label_offset = (10, -25)

            create_quadratic_graph(
                equation=equation,
                settings=settings,
            )

    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()