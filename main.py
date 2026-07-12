from generators.linear import create_linear_graph
from generators.quadratic import create_quadratic_graph


def main() -> None:
    print("Math Visual Generator")
    print("---------------------")
    print("1. Linear graph")
    print("2. Quadratic graph")

    choice = input("Choose a graph type: ")

    try:
        if choice == "1":
            equation = input(
                "Enter a linear expression, for example 2*x - 4: "
            )

            create_linear_graph(
                equation=equation,
                x_min=-10,
                x_max=10,
                output_name="linear_graph.png",
            )

        elif choice == "2":
            equation = input(
                "Enter a quadratic expression, for example "
                "x**2 - 4*x + 3: "
            )

            create_quadratic_graph(
                equation=equation,
                x_min=-10,
                x_max=10,
                output_name="quadratic_graph.png",
            )

        else:
            print("Invalid choice. Enter 1 or 2.")

    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()