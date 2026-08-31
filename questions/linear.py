from __future__ import annotations

import random
import logging
from pathlib import Path
from uuid import uuid4

from generators.api import generate_graph_from_request
from ai.question_writer import AIQuestionText, write_linear_question
from models.graph_request import GraphDisplaySettings, GraphRange, GraphRequest
from models.question_models import (
    GeneratedQuestion,
    LinearQuestionData,
    QuestionBatch,
    QuestionBlueprint,
)

logger = logging.getLogger(__name__)


def build_linear_display_settings(question_type: str) -> GraphDisplaySettings:
    display = GraphDisplaySettings()

    # General learner-facing style
    display.show_title = False
    display.show_legend = False
    display.show_equation = False

    display.show_grid = False
    display.show_border = False
    display.show_tick_labels = False
    display.show_tick_marks = False

    display.show_axes = True
    display.show_axis_arrows = True
    display.show_axis_labels = True
    display.show_origin_label = True

    if question_type == "determine_equation":
        display.show_x_intercepts = True
        display.show_y_intercepts = True

        display.show_gradient = False
        display.show_gradient_triangle = False

    elif question_type == "x_intercept":
        display.show_x_intercepts = False
        display.show_y_intercepts = True

    elif question_type == "y_intercept":
        display.show_y_intercepts = False
        display.show_x_intercepts = True

    elif question_type == "gradient":
        display.show_gradient = False
        display.show_gradient_triangle = False

        display.show_x_intercepts = True
        display.show_y_intercepts = True

    elif question_type == "find_f_of_x":
        display.show_x_intercepts = True
        display.show_y_intercepts = True

        display.show_gradient = False
        display.show_gradient_triangle = False

    elif question_type == "find_x_given_y":
        display.show_x_intercepts = True
        display.show_y_intercepts = True

        display.show_gradient = False
        display.show_gradient_triangle = False

    elif question_type == "read_coordinate":
        display.show_x_intercepts = False
        display.show_y_intercepts = False

        display.show_gradient = False
        display.show_gradient_triangle = False

        display.show_additional_point_labels = True

        # Needed to read point A from the axes
        display.show_grid = True
        display.show_tick_labels = True
        display.show_tick_marks = True

    elif question_type == "increasing_or_decreasing":
        display.show_x_intercepts = False
        display.show_y_intercepts = False

        display.show_gradient = False
        display.show_gradient_triangle = False

    else:
        raise ValueError(
            f"Unsupported linear question type: {question_type}"
        )

    return display


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

    return GraphRange(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def build_question_text(
    question_type: str,
    equation: str | None = None,
    input_x: int | float | None = None,
    target_y: int | float | None = None,
) -> str:
    """Build deterministic question text for a linear question type."""
    equation_text = (
        f"The graph of f(x) = {_display_equation_value(equation)} is shown below. "
        if equation
        else ""
    )

    if question_type == "x_intercept":
        return f"{equation_text}Determine the x-intercept of the graph."
    elif question_type == "y_intercept":
        return f"{equation_text}Determine the y-intercept of the graph."
    elif question_type == "gradient":
        return f"{equation_text}Determine the gradient of the graph."
    elif question_type == "determine_equation":
        return "Determine the equation of the linear graph shown below."
    elif question_type == "find_f_of_x":
        return f"The graph of f(x) = {_display_equation_value(equation)} is shown below. Determine f({_format_number(input_x)})."
    elif question_type == "find_x_given_y":
        return f"The graph of f(x) = {_display_equation_value(equation)} is shown below. Determine the value of x if f(x) = {_format_number(target_y)}."
    elif question_type == "read_coordinate":
        return "Write down the coordinates of point A."
    elif question_type == "increasing_or_decreasing":
        return "State whether the function shown below is increasing or decreasing."
    else:
        raise ValueError(f"Unsupported linear question type: {question_type}")


def _display_equation_value(equation: str) -> str:
    return equation.replace("*x", "x")


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
    else:
        raise ValueError(f"Unsupported linear question type: {question_type}")

    return answer, memo


def generate_linear_question_batch(
    blueprint: QuestionBlueprint,
    *,
    seed: int | None = None,
    use_ai: bool = False,
) -> QuestionBatch:

    blueprint.validate()
    batch_id = f"linear_{uuid4().hex}"
    batch_output_directory = Path("generated_graphs") / batch_id
    batch_output_directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    fingerprints: set = set()
    questions: list[GeneratedQuestion] = []
    max_attempts = blueprint.number_of_questions * 5
    attempts = 0

    while len(questions) < blueprint.number_of_questions and attempts < max_attempts:
        attempts += 1
        question_type = rng.choice(blueprint.question_types)
        data = _generate_data(rng, blueprint.difficulty)
        if question_type == "determine_equation" and data.y_intercept == 0:
            continue

        # Enhance data with type-specific fields
        data = _enhance_data_for_question_type(rng, data, question_type)

        # Create fingerprint that distinguishes all question types properly
        if question_type == "find_f_of_x":
            fingerprint = (question_type, data.gradient, data.y_intercept, data.input_x)
        elif question_type == "find_x_given_y":
            fingerprint = (question_type, data.gradient, data.y_intercept, data.target_y)
        elif question_type == "read_coordinate":
            fingerprint = (question_type, data.gradient, data.y_intercept, data.selected_point)
        else:
            fingerprint = (question_type, data.gradient, data.y_intercept)

        if fingerprint in fingerprints:
            continue

        display = build_linear_display_settings(question_type)

        # Add point to graph for read_coordinate questions
        if question_type == "read_coordinate" and data.selected_point:
            display.additional_x_values = [data.selected_point[0]]
            display.additional_point_labels = ["A"]

        request = GraphRequest(
            graph_type="Linear",
            equation=data.equation,
            graph_range=_select_range(data, question_type),
            display=display,
            output_name=f"linear_{len(questions) + 1:04d}.png",
        )

        # Build answer and memo with type-specific parameters
        answer, memo = build_memo(question_type, data)
        question_id = f"linear_{len(questions) + 1:04d}"
        question_text = build_question_text(
            question_type,
            equation=data.equation,
            input_x=data.input_x,
            target_y=data.target_y,
        )

        if use_ai:
            # Build visible/hidden information based on question type
            visible_information = []
            hidden_information = []

            if question_type == "x_intercept":
                visible_information = ["equation", "y-intercept"]
                hidden_information = ["x-intercept"]
            elif question_type == "y_intercept":
                visible_information = ["equation", "x-intercept"]
                hidden_information = ["y-intercept"]
            elif question_type == "gradient":
                visible_information = ["equation", "x-intercept", "y-intercept"]
                hidden_information = ["gradient"]
            elif question_type == "determine_equation":
                visible_information = ["x-intercept", "y-intercept"]
                hidden_information = ["equation", "gradient"]
            elif question_type == "find_f_of_x":
                visible_information = ["equation", "input_x"]
                hidden_information = ["f(x) value"]
            elif question_type == "find_x_given_y":
                visible_information = ["equation", "target_y"]
                hidden_information = ["x value"]
            elif question_type == "read_coordinate":
                visible_information = ["point label", "graph"]
                hidden_information = ["coordinates"]
            elif question_type == "increasing_or_decreasing":
                visible_information = ["graph"]
                hidden_information = ["gradient", "equation"]

            try:
                ai_text: AIQuestionText = write_linear_question(
                    grade=blueprint.grade,
                    difficulty=blueprint.difficulty,
                    question_type=question_type,
                    equation=data.equation,
                    expected_answer=answer,
                    gradient=data.gradient,
                    x_intercept=data.x_intercept,
                    y_intercept=data.y_intercept,
                    visible_information=visible_information,
                    hidden_information=hidden_information,
                    input_x=data.input_x,
                    target_y=data.target_y,
                )
                question_text, memo = ai_text.question_text, ai_text.memo
                logger.info("AI wording generated for %s", question_id)
            except Exception as e:
                logger.warning(
                    "AI request failed for %s; using deterministic fallback",
                    question_id,
                )
                logger.error(
                    "AI error for %s: %s: %s",
                    question_id,
                    type(e).__name__,
                    e,
                )

        artifact = generate_graph_from_request(
            request,
            output_directory=batch_output_directory,
        )
        questions.append(
            GeneratedQuestion(
                question_id=question_id,
                question_type=question_type,
                subject=blueprint.subject,
                grade=blueprint.grade,
                topic=blueprint.topic,
                subtopic=blueprint.subtopic,
                difficulty=blueprint.difficulty,
                marks=blueprint.marks_per_question,
                question_text=question_text,
                expected_answer=answer,
                memo=memo,
                mathematical_data=data,
                graph_request=request,
                graph_artifact=artifact,
            )
        )
        fingerprints.add(fingerprint)

    if len(questions) != blueprint.number_of_questions:
        raise ValueError(
            f"Could not generate {blueprint.number_of_questions} unique questions; "
            f"generated {len(questions)} after {attempts} attempts."
        )

    return QuestionBatch(
        blueprint=blueprint,
        questions=questions,
        batch_id=batch_id,
    )
