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
    """
    Build display settings that provide sufficient information for learners to answer the question.
    
    The policy ensures:
    - The question's requested answer is not directly visible (no cheating)
    - But enough information IS visible to solve it correctly
    
    x_intercept:
      - Hide the x-intercept marker/label (the answer itself)
      - Show the equation (OR in future: show two labeled points)
    
    y_intercept:
      - Hide the y-intercept marker/label (the answer itself)
      - Show the equation (OR in future: show enough points to determine y-intercept)
    
    gradient:
      - Hide gradient annotations and triangle
      - Show the equation (OR in future: show at least two labeled points)
    """
    if question_type not in {"x_intercept", "y_intercept", "gradient", "determine_equation"}:
        raise ValueError(f"Unsupported linear question type: {question_type}")
    
    display = GraphDisplaySettings()
    display.show_title = False
    display.show_legend = False
    display.show_equation = False

    if question_type == "determine_equation":
        return display
    
    if question_type == "x_intercept":
        # Hide the x-intercept marker/point label since that's the answer
        display.show_x_intercepts = False
        # Show y-intercept for reference, it helps visualize the line
        display.show_y_intercepts = True
    elif question_type == "y_intercept":
        # Hide the y-intercept marker/point label since that's the answer
        display.show_y_intercepts = False
        # Show x-intercept for reference, it helps visualize the line
        display.show_x_intercepts = True
    else:  # gradient
        # Hide gradient annotations and triangle since gradient is the answer
        display.show_gradient = False
        display.show_gradient_triangle = False
        # Show intercepts for reference points
        display.show_x_intercepts = True
        display.show_y_intercepts = True
    
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


def build_question_text(question_type: str, equation: str | None = None) -> str:
    equation_text = f"The graph of f(x) = {_display_equation_value(equation)} is shown below. " if equation else ""
    prompts = {
        "x_intercept": f"{equation_text}Determine the x-intercept of the graph.",
        "y_intercept": f"{equation_text}Determine the y-intercept of the graph.",
        "gradient": f"{equation_text}Determine the gradient of the graph.",
        "determine_equation": "Determine the equation of the linear graph shown below.",
    }
    try:
        return prompts[question_type]
    except KeyError as error:
        raise ValueError(f"Unsupported linear question type: {question_type}") from error


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
    else:  # gradient
        answer = gradient
        memo = (
            f"The gradient (slope) is the rate of change of the line.\n\n"
            f"From the equation displayed on the graph: y = {data.equation.replace('*', '')}\n"
            f"In the form y = mx + c, the gradient m is the coefficient of x.\n\n"
            f"Therefore, the gradient is {answer}."
        )
    
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
        answer, memo = build_memo(question_type, data)
        question_id = f"linear_{len(questions) + 1:04d}"
        question_text = build_question_text(question_type, data.equation)
        if use_ai:
            visible_information = ["equation"]
            hidden_information = [question_type]
            if question_type == "x_intercept":
                visible_information.append("y-intercept")
            elif question_type == "y_intercept":
                visible_information.append("x-intercept")
            else:
                visible_information.extend(["x-intercept", "y-intercept"])
                hidden_information.append("gradient")
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
