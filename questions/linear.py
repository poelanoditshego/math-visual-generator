from __future__ import annotations

import random
from pathlib import Path

from generators.api import generate_graph_from_request
from models.graph_request import GraphDisplaySettings, GraphRange, GraphRequest
from models.question_models import (
    GeneratedQuestion,
    LinearQuestionData,
    QuestionBatch,
    QuestionBlueprint,
)


def build_linear_display_settings(question_type: str) -> GraphDisplaySettings:
    if question_type not in {"x_intercept", "y_intercept", "gradient"}:
        raise ValueError(f"Unsupported linear question type: {question_type}")
    display = GraphDisplaySettings()
    display.show_equation = False
    if question_type == "x_intercept":
        display.show_x_intercepts = False
    elif question_type == "y_intercept":
        display.show_y_intercepts = False
    else:
        display.show_gradient = False
        display.show_gradient_triangle = False
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


def _select_range(data: LinearQuestionData) -> GraphRange:
    x_intercept = int(data.x_intercept or 0)
    x_min = min(0, x_intercept) - 3
    x_max = max(0, x_intercept) + 3
    left_y = data.gradient * x_min + data.y_intercept
    right_y = data.gradient * x_max + data.y_intercept
    y_min = min(0, data.y_intercept, left_y, right_y) - 3
    y_max = max(0, data.y_intercept, left_y, right_y) + 3
    return GraphRange(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def build_question_text(question_type: str) -> str:
    prompts = {
        "x_intercept": "Determine the x-intercept of the linear graph shown below.",
        "y_intercept": "Determine the y-intercept of the linear graph shown below.",
        "gradient": "Determine the gradient of the linear graph shown below.",
    }
    try:
        return prompts[question_type]
    except KeyError as error:
        raise ValueError(f"Unsupported linear question type: {question_type}") from error


def build_memo(question_type: str, data: LinearQuestionData) -> tuple[str, str]:
    gradient = _format_number(data.gradient)
    intercept = _format_number(data.y_intercept)
    if question_type == "x_intercept":
        answer = f"({_format_number(data.x_intercept or 0)}, 0)"
        memo = (
            f"At the x-intercept, y = 0.\n\n"
            f"0 = {data.equation.replace('*', '')}\n"
            f"Therefore, the x-intercept is {answer}."
        )
    elif question_type == "y_intercept":
        answer = f"(0, {intercept})"
        memo = (
            "At the y-intercept, x = 0.\n\n"
            f"f(0) = {data.equation.replace('*', '')}\n"
            f"Therefore, the y-intercept is {answer}."
        )
    else:
        answer = gradient
        memo = (
            "For a linear function in the form y = mx + c, "
            f"the coefficient of x is {gradient}.\n\n"
            f"Therefore, the gradient is {answer}."
        )
    return answer, memo


def generate_linear_question_batch(
    blueprint: QuestionBlueprint,
    *,
    seed: int | None = None,
) -> QuestionBatch:
    blueprint.validate()
    rng = random.Random(seed)
    fingerprints: set[tuple[str, int | float, int | float]] = set()
    questions: list[GeneratedQuestion] = []
    max_attempts = blueprint.number_of_questions * 5
    attempts = 0

    while len(questions) < blueprint.number_of_questions and attempts < max_attempts:
        attempts += 1
        question_type = rng.choice(blueprint.question_types)
        data = _generate_data(rng, blueprint.difficulty)
        fingerprint = (question_type, data.gradient, data.y_intercept)
        if fingerprint in fingerprints:
            continue

        display = build_linear_display_settings(question_type)
        request = GraphRequest(
            graph_type="Linear",
            equation=data.equation,
            graph_range=_select_range(data),
            display=display,
            output_name=f"linear_{len(questions) + 1:04d}.png",
        )
        artifact = generate_graph_from_request(request)
        answer, memo = build_memo(question_type, data)
        question_id = f"linear_{len(questions) + 1:04d}"
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
                question_text=build_question_text(question_type),
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
        batch_id="linear_batch_" + str(seed if seed is not None else "random"),
    )
