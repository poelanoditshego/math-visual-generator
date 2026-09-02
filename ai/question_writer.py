from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ai.client import create_structured_response


@dataclass(frozen=True)
class AIQuestionText:
    question_text: str
    memo: str


def _display_equation(equation: str) -> str:
    return equation.replace("*x", "x")


def normalize_answer(value: str) -> str:
    return value.lower().replace(" ", "").replace(";", ",").replace(".", "")


def _contains_verified_answer(text: str, expected_answer: str, question_type: str) -> bool:
    """Reject answer disclosure without mistaking visible equation data for an answer."""
    normalized_text = normalize_answer(text)
    normalized_answer = normalize_answer(expected_answer)
    if question_type == "gradient":
        return False
    if question_type == "increasing_or_decreasing":
        answer = expected_answer.lower()
        return bool(re.search(rf"\b(?:is|is\s+shown\s+as)\s+{answer}\b", text.lower()))
    if question_type == "read_coordinate":
        return normalized_answer in normalized_text
    if question_type == "find_f_of_x":
        return bool(re.search(rf"f\s*\([^)]*\)\s*=\s*{re.escape(expected_answer)}\b", text, re.I))
    if question_type == "find_x_given_y":
        return bool(re.search(rf"\bx\s*=\s*{re.escape(expected_answer)}\b", text, re.I))
    return normalized_answer in normalized_text


_QUESTION_TYPE_PHRASES = {
    "x_intercept": ("x-intercept", "x intercept", "x-axis"),
    "y_intercept": ("y-intercept", "y intercept", "y-axis"),
    "gradient": ("gradient", "slope"),
    "determine_equation": (
        "determine the equation", "find the equation", "write down the equation", "equation of the",
    ),
    "find_f_of_x": ("determine f(", "calculate f(", "find f(", "value of f("),
    "find_x_given_y": ("determine the value of x", "find x", "determine x"),
    "read_coordinate": ("coordinates of point", "coordinate of point", "coordinates of"),
    "increasing_or_decreasing": ("increasing or decreasing", "state whether", "classify"),
    "intersection_of_two_lines": (
        "point of intersection", "coordinates of the intersection", "intersect",
    ),
}


def _compact_equation(value: str) -> str:
    return _display_equation(value).replace(" ", "").replace("*", "").lower()


def _validate_response(
    payload: object,
    *,
    question_type: str,
    equation: str,
    expected_answer: str,
    visible_information: list[str],
    hidden_information: list[str],
) -> AIQuestionText:
    if not isinstance(payload, dict) or set(payload) != {"question_text", "memo"}:
        raise ValueError("AI response must contain exactly question_text and memo")
    question_text, memo = payload["question_text"], payload["memo"]
    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError("AI question_text must be a non-empty string")
    if not isinstance(memo, str) or not memo.strip():
        raise ValueError("AI memo must be a non-empty string")
    if "todo" in question_text.lower() or "todo" in memo.lower():
        raise ValueError("AI response contains a placeholder")
    if normalize_answer(expected_answer) not in normalize_answer(memo):
        raise ValueError("AI memo must include the verified answer")
    if _contains_verified_answer(question_text, expected_answer, question_type):
        raise ValueError("AI question_text reveals the verified answer")

    compact_question = _compact_equation(question_text)
    compact_equation = _compact_equation(equation)
    if "equation" in visible_information and compact_equation not in compact_question:
        raise ValueError("AI question_text omitted the visible equation")
    if "equation" in hidden_information and compact_equation in compact_question:
        raise ValueError("AI question_text reveals hidden equation")
    if "gradient" in hidden_information and re.search(r"\bgradient\s*[:=]", question_text, re.I):
        raise ValueError("AI question_text reveals hidden gradient information")

    phrases = _QUESTION_TYPE_PHRASES.get(question_type)
    if not phrases:
        raise ValueError(f"Unsupported question type for AI validation: {question_type}")
    if not any(phrase in question_text.lower() for phrase in phrases):
        raise ValueError("AI question_text does not match the requested type")
    return AIQuestionText(question_text=question_text.strip(), memo=memo.strip())


def _prompts(
    *, grade: int, difficulty: str, question_type: str, equation: str,
    expected_answer: str, gradient: int | float, x_intercept: int | float | None,
    y_intercept: int | float, visible_information: list[str], hidden_information: list[str],
    input_x: int | float | None = None, target_y: int | float | None = None,
    second_equation: str | None = None,
) -> tuple[str, str]:
    system_prompt = (
        "You write South African school Mathematics questions. The mathematics supplied "
        "is verified and must not be changed. Return only the requested JSON object. "
        "Use only visible information in the learner-facing question and never reveal "
        "hidden information. The memo must explain the method and end with the verified answer exactly."
    )
    payload: dict[str, object] = {
        "grade": grade, "difficulty": difficulty, "question_type": question_type,
        "equation": equation, "verified_answer": expected_answer, "gradient": gradient,
        "x_intercept": x_intercept, "y_intercept": y_intercept,
        "visible_information": visible_information, "hidden_information": hidden_information,
        "output_requirements": {
            "question_text": "One learner-facing question. Include the equation only when it is visible.",
            "memo": "A concise worked method ending with the verified answer.",
        },
    }
    if input_x is not None:
        payload["input_x"] = input_x
    if target_y is not None:
        payload["target_y"] = target_y
    if second_equation is not None:
        payload["second_equation"] = second_equation
    return system_prompt, json.dumps(payload, indent=2)


def write_linear_question(
    *, grade: int, difficulty: str, question_type: str, equation: str,
    expected_answer: str, gradient: int | float, x_intercept: int | float | None,
    y_intercept: int | float, visible_information: list[str], hidden_information: list[str],
    input_x: int | float | None = None, target_y: int | float | None = None,
    second_equation: str | None = None,
) -> AIQuestionText:
    """Generate and validate AI wording, retrying one invalid response."""
    system_prompt, user_prompt = _prompts(
        grade=grade, difficulty=difficulty, question_type=question_type, equation=equation,
        expected_answer=expected_answer, gradient=gradient, x_intercept=x_intercept,
        y_intercept=y_intercept, visible_information=visible_information,
        hidden_information=hidden_information, input_x=input_x, target_y=target_y,
        second_equation=second_equation,
    )
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = create_structured_response(system_prompt=system_prompt, user_prompt=user_prompt)
            return _validate_response(
                json.loads(response.output_text), question_type=question_type,
                equation=equation, expected_answer=expected_answer,
                visible_information=visible_information, hidden_information=hidden_information,
            )
        except Exception as error:
            last_error = error
    raise ValueError("AI response failed validation after one retry") from last_error
