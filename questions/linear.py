from __future__ import annotations

import random
from ai.question_writer import AIQuestionText, write_linear_question
from models.graph_request import GraphDisplaySettings, GraphRange, GraphRequest
from models.question_models import (
    LinearQuestionData,
    QuestionBatch,
    QuestionBlueprint,
)
from questions.engine import (
    QuestionCandidate,
    generate_question_batch,
    register_family_generator,
)
from questions.specs import QuestionSpec, get_question_spec


def build_linear_display_settings(question_type: str) -> GraphDisplaySettings:
    """Return the registered learner-facing display policy for a Linear type."""
    return get_question_spec(question_type, family="linear").build_display_settings()


def _format_number(value: int | float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_linear_equation(gradient: int, y_intercept: int) -> str:
    if gradient == 1:
        gradient_text = "x"
    elif gradient == -1:
        gradient_text = "-x"
    else:
        gradient_text = f"{gradient}*x"
    if y_intercept == 0:
        return gradient_text
    sign = "+" if y_intercept > 0 else "-"
    return f"{gradient_text} {sign} {abs(y_intercept)}"


def _display_equation(data: LinearQuestionData) -> str:
    equation = data.equation.replace("*x", "x")
    return equation.replace(" + ", " + ").replace(" - ", " - ")


def _candidate_parameters(difficulty: str) -> list[tuple[int, int, int]]:
    if difficulty == "Easy":
        gradients = [-2, -1, 1, 2]
        roots = range(-5, 6)
    elif difficulty == "Medium":
        gradients = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        roots = range(-5, 6)
    else:
        gradients = [-5, -4, -3, -2, -1, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        roots = range(-8, 9)

    candidates = []
    for gradient in gradients:
        for x_intercept in roots:
            y_intercept = -gradient * x_intercept
            if -10 <= y_intercept <= 10:
                candidates.append((gradient, y_intercept, x_intercept))
    return candidates


def _generate_data(rng: random.Random, difficulty: str) -> LinearQuestionData:
    gradient, y_intercept, x_intercept = rng.choice(
        _candidate_parameters(difficulty)
    )
    return LinearQuestionData(
        equation=_format_linear_equation(gradient, y_intercept),
        gradient=gradient,
        y_intercept=y_intercept,
        x_intercept=x_intercept,
    )


def _generate_intersection_data(
    rng: random.Random,
    difficulty: str,
) -> LinearQuestionData:
    """Construct two distinct lines through one clean integer intersection."""
    gradient_limit = {"Easy": 2, "Medium": 4}.get(difficulty, 5)
    gradients = [
        value for value in range(-gradient_limit, gradient_limit + 1) if value
    ]
    coordinate_limit = 3 if difficulty == "Easy" else 5
    coordinates = [
        value for value in range(-coordinate_limit, coordinate_limit + 1) if value
    ]
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for intersection_x in coordinates:
        for intersection_y in coordinates:
            for first_index, first_gradient in enumerate(gradients):
                for second_gradient in gradients[first_index + 1:]:
                    first_intercept = intersection_y - first_gradient * intersection_x
                    second_intercept = intersection_y - second_gradient * intersection_x
                    if abs(first_intercept) <= 10 and abs(second_intercept) <= 10:
                        candidates.append(
                            (
                                intersection_x,
                                intersection_y,
                                second_gradient,
                                first_gradient,
                                second_intercept,
                                first_intercept,
                            )
                        )

    (
        intersection_x,
        intersection_y,
        first_gradient,
        second_gradient,
        first_intercept,
        second_intercept,
    ) = rng.choice(candidates)
    return LinearQuestionData(
        equation=_format_linear_equation(first_gradient, first_intercept),
        gradient=first_gradient,
        y_intercept=first_intercept,
        x_intercept=-first_intercept / first_gradient,
        second_equation=_format_linear_equation(second_gradient, second_intercept),
        second_gradient=second_gradient,
        second_y_intercept=second_intercept,
        intersection_point=(intersection_x, intersection_y),
    )


def _generate_two_point_data(
    rng: random.Random,
    difficulty: str,
) -> LinearQuestionData:
    """Construct an integer line and two readable, non-axis points on it."""
    if difficulty == "Easy":
        gradient_limit, intercept_limit, x_limit, y_limit = 2, 4, 4, 8
    elif difficulty == "Medium":
        gradient_limit, intercept_limit, x_limit, y_limit = 4, 6, 5, 12
    else:
        gradient_limit, intercept_limit, x_limit, y_limit = 5, 8, 6, 16

    candidates: list[tuple[int, int, int, int]] = []
    gradients = [value for value in range(-gradient_limit, gradient_limit + 1) if value]
    for gradient in gradients:
        for y_intercept in range(-intercept_limit, intercept_limit + 1):
            valid_x_values = [
                x_value
                for x_value in range(-x_limit, x_limit + 1)
                if x_value != 0
                and gradient * x_value + y_intercept != 0
                and abs(gradient * x_value + y_intercept) <= y_limit
            ]
            for first_index, x1 in enumerate(valid_x_values):
                for x2 in valid_x_values[first_index + 1:]:
                    if abs(x2 - x1) >= 2:
                        candidates.append((gradient, y_intercept, x1, x2))

    gradient, y_intercept, x1, x2 = rng.choice(candidates)
    point_a = (x1, gradient * x1 + y_intercept)
    point_b = (x2, gradient * x2 + y_intercept)
    return LinearQuestionData(
        equation=_format_linear_equation(gradient, y_intercept),
        gradient=gradient,
        y_intercept=y_intercept,
        x_intercept=-y_intercept / gradient,
        point_a=point_a,
        point_b=point_b,
    )


def _generate_gradient_point_data(
    rng: random.Random,
    difficulty: str,
) -> LinearQuestionData:
    """Construct an integer line and one readable non-axis point on it."""
    if difficulty == "Easy":
        gradient_limit, intercept_limit, x_limit, y_limit = 2, 4, 4, 8
    elif difficulty == "Medium":
        gradient_limit, intercept_limit, x_limit, y_limit = 4, 6, 5, 12
    else:
        gradient_limit, intercept_limit, x_limit, y_limit = 5, 8, 6, 16

    candidates: list[tuple[int, int, int]] = []
    gradients = [value for value in range(-gradient_limit, gradient_limit + 1) if value]
    for gradient in gradients:
        for y_intercept in range(-intercept_limit, intercept_limit + 1):
            for x_value in range(-x_limit, x_limit + 1):
                y_value = gradient * x_value + y_intercept
                if x_value != 0 and y_value != 0 and abs(y_value) <= y_limit:
                    candidates.append((gradient, y_intercept, x_value))

    gradient, y_intercept, x_value = rng.choice(candidates)
    return LinearQuestionData(
        equation=_format_linear_equation(gradient, y_intercept),
        gradient=gradient,
        y_intercept=y_intercept,
        x_intercept=-y_intercept / gradient,
        point_a=(x_value, gradient * x_value + y_intercept),
    )


def _enhance_data_for_question_type(
    rng: random.Random,
    data: LinearQuestionData,
    question_type: str,
) -> LinearQuestionData:
    """
    Enhance LinearQuestionData with type-specific fields.

    For new question types, this generates and stores:
    - input_x: the x-value for find_f_of_x and find_x_given_y
    - target_y: the y-value target for find_x_given_y
    - selected_point: the point to display for read_coordinate
    """
    candidate_x_values = [
        value for value in range(-3, 4)
        if abs(data.gradient * value + data.y_intercept) <= 12
    ]
    if not candidate_x_values:
        candidate_x_values = [0]

    if question_type == "find_f_of_x":
        # Keep both the requested input and its output learner-friendly.
        input_x = rng.choice(candidate_x_values)
        return LinearQuestionData(
            equation=data.equation,
            gradient=data.gradient,
            y_intercept=data.y_intercept,
            x_intercept=data.x_intercept,
            function_name=data.function_name,
            input_x=input_x,
            target_y=None,
            selected_point=None,
        )

    elif question_type == "find_x_given_y":
        # Select an x-value, then calculate corresponding y
        # This ensures the answer is exact
        input_x = rng.choice(candidate_x_values)
        target_y = data.gradient * input_x + data.y_intercept
        return LinearQuestionData(
            equation=data.equation,
            gradient=data.gradient,
            y_intercept=data.y_intercept,
            x_intercept=data.x_intercept,
            function_name=data.function_name,
            input_x=input_x,
            target_y=target_y,
            selected_point=None,
        )

    elif question_type == "read_coordinate":
        # Select an integer point that lies on the line
        # Avoid points too close to boundaries
        non_axis_candidates = [
            value for value in candidate_x_values
            if value != 0 and data.gradient * value + data.y_intercept != 0
        ]
        selected_x = rng.choice(non_axis_candidates or candidate_x_values)
        selected_y = data.gradient * selected_x + data.y_intercept
        return LinearQuestionData(
            equation=data.equation,
            gradient=data.gradient,
            y_intercept=data.y_intercept,
            x_intercept=data.x_intercept,
            function_name=data.function_name,
            input_x=None,
            target_y=None,
            selected_point=(selected_x, selected_y),
        )

    else:
        # For existing question types, no enhancement needed
        return data


def _select_range(data: LinearQuestionData, question_type: str) -> GraphRange:
    """
    Select appropriate graph range based on question type.

    For new types with specific points (find_f_of_x, find_x_given_y, read_coordinate),
    ensure those points are visible with padding.
    """
    if question_type == "intersection_of_two_lines" and data.intersection_point:
        intersection_x, intersection_y = data.intersection_point
        x_min = min(0, intersection_x) - 4
        x_max = max(0, intersection_x) + 4
        endpoints = [
            data.gradient * x_value + data.y_intercept
            for x_value in (x_min, x_max)
        ]
        if data.second_gradient is not None and data.second_y_intercept is not None:
            endpoints.extend(
                data.second_gradient * x_value + data.second_y_intercept
                for x_value in (x_min, x_max)
            )
        return GraphRange(
            x_min=x_min,
            x_max=x_max,
            y_min=min(0, intersection_y, *endpoints) - 2,
            y_max=max(0, intersection_y, *endpoints) + 2,
        )

    x_intercept = int(data.x_intercept or 0)
    x_min = min(0, x_intercept) - 3
    x_max = max(0, x_intercept) + 3
    left_y = data.gradient * x_min + data.y_intercept
    right_y = data.gradient * x_max + data.y_intercept
    y_min = min(0, data.y_intercept, left_y, right_y) - 3
    y_max = max(0, data.y_intercept, left_y, right_y) + 3

    # For types with specific points, ensure they're visible with padding
    if question_type == "find_f_of_x" and data.input_x is not None:
        x_val = data.input_x
        y_val = data.gradient * x_val + data.y_intercept
        x_min = min(x_min, x_val - 1)
        x_max = max(x_max, x_val + 1)
        y_min = min(y_min, y_val - 1)
        y_max = max(y_max, y_val + 1)

    elif question_type == "find_x_given_y" and data.input_x is not None:
        x_val = data.input_x
        y_val = data.target_y or 0
        x_min = min(x_min, x_val - 1)
        x_max = max(x_max, x_val + 1)
        y_min = min(y_min, y_val - 1)
        y_max = max(y_max, y_val + 1)

    elif question_type == "read_coordinate" and data.selected_point:
        x_val, y_val = data.selected_point
        x_min = min(x_min, x_val - 1)
        x_max = max(x_max, x_val + 1)
        y_min = min(y_min, y_val - 1)
        y_max = max(y_max, y_val + 1)

    elif question_type == "equation_from_two_points" and data.point_a and data.point_b:
        x_values = (data.point_a[0], data.point_b[0])
        x_min = min(0, *x_values) - 2
        x_max = max(0, *x_values) + 2
        y_values = (
            data.point_a[1],
            data.point_b[1],
            data.gradient * x_min + data.y_intercept,
            data.gradient * x_max + data.y_intercept,
        )
        y_min = min(0, *y_values) - 2
        y_max = max(0, *y_values) + 2

    elif question_type == "equation_from_gradient_and_point" and data.point_a:
        x_value, y_value = data.point_a
        x_min = min(0, x_value) - 2
        x_max = max(0, x_value) + 2
        endpoint_y_values = (
            data.gradient * x_min + data.y_intercept,
            data.gradient * x_max + data.y_intercept,
        )
        y_min = min(0, y_value, *endpoint_y_values) - 2
        y_max = max(0, y_value, *endpoint_y_values) + 2

    return GraphRange(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def build_question_text(
    question_type: str,
    equation: str | None = None,
    input_x: int | float | None = None,
    target_y: int | float | None = None,
    gradient: int | float | None = None,
    point_a: tuple[int | float, int | float] | None = None,
    point_b: tuple[int | float, int | float] | None = None,
) -> str:
    """Build deterministic question text for a linear question type."""
    equation_text = (
        f"The graph of f(x) = {_display_equation_value(equation)} is shown below. "
        if equation
        else ""
    )

    spec = get_question_spec(question_type, family="linear")
    return spec.question_template.format(
        equation_prefix=equation_text,
        equation=_display_equation_value(equation) if equation else "",
        input_x=_format_number(input_x or 0),
        target_y=_format_number(target_y or 0),
        gradient=_format_number(gradient or 0),
        point_a=_format_point(point_a) if point_a else "",
        point_b=_format_point(point_b) if point_b else "",
    )


def _display_equation_value(equation: str) -> str:
    return equation.replace("*x", "x")


def _format_point(point: tuple[int | float, int | float]) -> str:
    return f"({_format_number(point[0])}; {_format_number(point[1])})"


def _format_subtraction(first: int | float, second: int | float) -> str:
    second_text = _format_number(second)
    if second < 0:
        second_text = f"({second_text})"
    return f"{_format_number(first)} - {second_text}"


def build_memo(question_type: str, data: LinearQuestionData) -> tuple[str, str]:
    """
    Build the memo (worked solution) for a Linear question.

    The memo explains how to answer using the GRAPH-BASED method, since the learner
    sees the graph and equation, not the underlying mathematical data.
    """
    gradient = _format_number(data.gradient)
    intercept = _format_number(data.y_intercept)

    if question_type == "x_intercept":
        answer = f"({_format_number(data.x_intercept or 0)}, 0)"
        memo = (
            f"The x-intercept is where the line crosses the x-axis (where y = 0).\n\n"
            f"From the equation displayed on the graph:\n"
            f"0 = {data.equation.replace('*', '')}\n"
            f"Solving for x: x = {_format_number(data.x_intercept or 0)}\n\n"
            f"Therefore, the x-intercept is {answer}."
        )
    elif question_type == "y_intercept":
        answer = f"(0, {intercept})"
        memo = (
            f"The y-intercept is where the line crosses the y-axis (where x = 0).\n\n"
            f"From the equation displayed on the graph:\n"
            f"f(0) = {data.equation.replace('*', '')}\n"
            f"f(0) = {intercept}\n\n"
            f"Therefore, the y-intercept is {answer}."
        )
    elif question_type == "gradient":
        answer = gradient
        memo = (
            f"The gradient (slope) is the rate of change of the line.\n\n"
            f"From the equation displayed on the graph: y = {data.equation.replace('*', '')}\n"
            f"In the form y = mx + c, the gradient m is the coefficient of x.\n\n"
            f"Therefore, the gradient is {answer}."
        )
    elif question_type == "determine_equation":
        equation = _display_equation(data)

        answer = f"f(x) = {equation}"

        memo = (
            f"From the graph, the x-intercept is "
            f"({_format_number(data.x_intercept or 0)}, 0) "
            f"and the y-intercept is (0, {intercept}).\n\n"
            f"The gradient is m = {gradient}.\n"
            f"The y-intercept gives c = {intercept}.\n\n"
            f"Using f(x) = mx + c:\n"
            f"f(x) = {equation}\n\n"
            f"Therefore, the equation is {answer}."
        )

    elif question_type == "equation_from_two_points":
        if data.point_a is None or data.point_b is None:
            raise ValueError("Two-point equation data is incomplete.")
        x1, y1 = data.point_a
        x2, y2 = data.point_b
        delta_y = y2 - y1
        delta_x = x2 - x1
        equation = _display_equation(data)
        answer = f"y = {equation}"
        memo = (
            f"A{_format_point(data.point_a)}\n"
            f"B{_format_point(data.point_b)}\n\n"
            "First determine the gradient:\n\n"
            f"m = ({_format_subtraction(y2, y1)}) / "
            f"({_format_subtraction(x2, x1)})\n"
            f"m = {_format_number(delta_y)} / {_format_number(delta_x)}\n"
            f"m = {gradient}\n\n"
            "Use y = mx + c.\n\n"
            f"Substitute A{_format_point(data.point_a)}:\n\n"
            f"{_format_number(y1)} = {gradient}({_format_number(x1)}) + c\n"
            f"c = {intercept}\n\n"
            f"Therefore, {answer}."
        )

    elif question_type == "equation_from_gradient_and_point":
        if data.point_a is None:
            raise ValueError("Gradient-and-point equation data is incomplete.")
        x_value, y_value = data.point_a
        product = data.gradient * x_value
        equation = _display_equation(data)
        answer = f"y = {equation}"
        memo = (
            "The gradient is:\n\n"
            f"m = {gradient}\n\n"
            "The line passes through:\n\n"
            f"A{_format_point(data.point_a)}\n\n"
            "Substitute into y = mx + c:\n\n"
            f"{_format_number(y_value)} = {gradient}({_format_number(x_value)}) + c\n"
            f"{_format_number(y_value)} = {_format_number(product)} + c\n"
            f"c = {intercept}\n\n"
            f"Therefore, {answer}."
        )

    elif question_type == "find_f_of_x":
        input_x = _format_number(data.input_x or 0)
        answer = _format_number(data.gradient * (data.input_x or 0) + data.y_intercept)
        memo = (
            f"To find f({input_x}), substitute x = {input_x} into the equation.\n\n"
            f"f({input_x}) = {data.equation.replace('*', '')} where x = {input_x}\n"
            f"f({input_x}) = {_format_number(data.gradient)}({input_x}) + {intercept}\n"
            f"f({input_x}) = {_format_number(data.gradient * (data.input_x or 0))} + {intercept}\n"
            f"f({input_x}) = {answer}\n\n"
            f"Therefore, f({input_x}) = {answer}."
        )

    elif question_type == "find_x_given_y":
        target_y = _format_number(data.target_y or 0)
        answer = _format_number(data.input_x or 0)
        memo = (
            f"To find x when f(x) = {target_y}, solve the equation:\n\n"
            f"{target_y} = {data.equation.replace('*', '')}\n"
            f"{target_y} = {_format_number(data.gradient)}x + {intercept}\n"
            f"{_format_number((data.target_y or 0) - data.y_intercept)} = {_format_number(data.gradient)}x\n"
            f"x = {answer}\n\n"
            f"Therefore, x = {answer}."
        )
    elif question_type == "read_coordinate":
        if data.selected_point:
            x_coord = _format_number(data.selected_point[0])
            y_coord = _format_number(data.selected_point[1])
            answer = f"({x_coord}; {y_coord})"
            memo = (
                f"Read the coordinates of point A from the graph.\n\n"
                f"From the x-axis: x = {x_coord}\n"
                f"From the y-axis: y = {y_coord}\n\n"
                f"Therefore, the coordinates of point A are ({x_coord}; {y_coord})."
            )
        else:
            answer = "(0; 0)"
            memo = "Point A is not properly defined."
    elif question_type == "increasing_or_decreasing":
        if data.gradient > 0:
            answer = "Increasing"
            memo = (
                f"The function has gradient m = {gradient}.\n\n"
                f"Since m = {gradient} > 0, the line rises from left to right.\n\n"
                f"Therefore, the function is increasing."
            )
        else:
            answer = "Decreasing"
            memo = (
                f"The function has gradient m = {gradient}.\n\n"
                f"Since m = {gradient} < 0, the line falls from left to right.\n\n"
                f"Therefore, the function is decreasing."
            )
    elif question_type == "intersection_of_two_lines":
        if (
            data.second_equation is None
            or data.second_gradient is None
            or data.second_y_intercept is None
            or data.intersection_point is None
        ):
            raise ValueError("Two-line intersection data is incomplete.")
        intersection_x, intersection_y = data.intersection_point
        first_equation = _display_equation(data)
        second_equation = data.second_equation.replace("*x", "x")
        gradient_difference = data.gradient - data.second_gradient
        intercept_difference = data.second_y_intercept - data.y_intercept
        substitution_sign = "+" if data.y_intercept >= 0 else "-"
        answer = (
            f"({_format_number(intersection_x)}; "
            f"{_format_number(intersection_y)})"
        )
        memo = (
            "At the point of intersection, f(x) = g(x).\n\n"
            f"{first_equation} = {second_equation}\n"
            f"{_format_number(gradient_difference)}x = "
            f"{_format_number(intercept_difference)}\n"
            f"x = {_format_number(intersection_x)}\n\n"
            "Substitute this value into f(x):\n"
            f"y = {_format_number(data.gradient)}"
            f"({_format_number(intersection_x)}) {substitution_sign} "
            f"{_format_number(abs(data.y_intercept))}\n"
            f"y = {_format_number(intersection_y)}\n\n"
            f"Therefore, the point of intersection is {answer}."
        )
    else:
        raise ValueError(f"Unsupported linear question type: {question_type}")

    return answer, memo


class LinearQuestionGenerator:
    """Linear-owned mathematics and rendering inputs for the generic engine."""

    family = "linear"
    batch_prefix = "linear"

    def create_candidate(
        self,
        *,
        rng: random.Random,
        difficulty: str,
        spec: QuestionSpec,
        output_name: str,
    ) -> QuestionCandidate | None:
        question_type = spec.question_type
        if question_type == "intersection_of_two_lines":
            data = _generate_intersection_data(rng, difficulty)
        elif question_type == "equation_from_two_points":
            data = _generate_two_point_data(rng, difficulty)
        elif question_type == "equation_from_gradient_and_point":
            data = _generate_gradient_point_data(rng, difficulty)
        else:
            data = _generate_data(rng, difficulty)
        if question_type == "determine_equation" and data.y_intercept == 0:
            return None
        data = _enhance_data_for_question_type(rng, data, question_type)
        fingerprint = (
            question_type,
            *(getattr(data, field) for field in spec.fingerprint_fields),
        )
        display = build_linear_display_settings(question_type)
        if question_type == "read_coordinate" and data.selected_point:
            display.additional_x_values = [data.selected_point[0]]
            display.additional_point_labels = ["A"]
        elif question_type == "equation_from_two_points" and data.point_a and data.point_b:
            display.additional_x_values = [data.point_a[0], data.point_b[0]]
            display.additional_point_labels = ["A", "B"]
        elif question_type == "equation_from_gradient_and_point" and data.point_a:
            display.additional_x_values = [data.point_a[0]]
            display.additional_point_labels = ["A"]
        is_two_line_question = question_type == "intersection_of_two_lines"
        graph_request = GraphRequest(
            graph_type="Mixed" if is_two_line_question else "Linear",
            equation=None if is_two_line_question else data.equation,
            equations=(
                [data.equation, data.second_equation]
                if is_two_line_question and data.second_equation
                else None
            ),
            graph_range=_select_range(data, question_type),
            display=display,
            output_name=output_name,
        )
        answer, memo = build_memo(question_type, data)
        return QuestionCandidate(
            mathematical_data=data,
            fingerprint=fingerprint,
            graph_request=graph_request,
            question_text=build_question_text(
                question_type,
                equation=data.equation,
                input_x=data.input_x,
                target_y=data.target_y,
                gradient=data.gradient,
                point_a=data.point_a,
                point_b=data.point_b,
            ),
            expected_answer=answer,
            memo=memo,
        )

    def rewrite_with_ai(
        self,
        *,
        blueprint: QuestionBlueprint,
        spec: QuestionSpec,
        candidate: QuestionCandidate,
    ) -> tuple[str, str]:
        data: LinearQuestionData = candidate.mathematical_data
        ai_text: AIQuestionText = write_linear_question(
            grade=blueprint.grade,
            difficulty=blueprint.difficulty,
            question_type=spec.question_type,
            equation=data.equation,
            expected_answer=candidate.expected_answer,
            gradient=data.gradient,
            x_intercept=data.x_intercept,
            y_intercept=data.y_intercept,
            visible_information=list(spec.ai_visible_information),
            hidden_information=list(spec.ai_hidden_information),
            input_x=data.input_x,
            target_y=data.target_y,
            second_equation=data.second_equation,
            point_a=data.point_a,
            point_b=data.point_b,
        )
        return ai_text.question_text, ai_text.memo


LINEAR_QUESTION_GENERATOR = LinearQuestionGenerator()
register_family_generator(LINEAR_QUESTION_GENERATOR)


def generate_linear_question_batch(
    blueprint: QuestionBlueprint,
    *,
    seed: int | None = None,
    use_ai: bool = False,
) -> QuestionBatch:
    """Backwards-compatible Linear wrapper around the generic engine."""
    return generate_question_batch(
        blueprint,
        seed=seed,
        use_ai=use_ai,
        family_generator=LINEAR_QUESTION_GENERATOR,
    )
