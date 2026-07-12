from models.graph_settings import GraphSettings


def main() -> None:
    settings = GraphSettings(
        x_min=-10,
        x_max=10,
        y_min=-5,
        y_max=15,
        title="My First Custom Graph",
        show_grid=True,
        show_intercepts=True,
    )

    print(settings)


if __name__ == "__main__":
    main()