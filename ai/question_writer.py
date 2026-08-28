from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ai.client import create_structured_response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIQuestionText:
    question_text: str
    memo: str


def _display_equation(equation: str) -> str:
    return equation.replace("*x", "x")


def _contains_verified_answer(text: str, expected_answer: str, question_type: str) -> bool:
    if question_type == "gradient":
        return False
    compact_text = text.replace(" ", "")
    compact_answer = expected_answer.replace(" ", "")
    alternate_answer = compact_answer.replace(",", ";")
    return compact_answer in compact_text or alternate_answer in compact_text

def normalize_answer(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "")
        .replace(";", ",")
        .replace(".", "")
    )





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
    question_text = payload["question_text"]
    memo = payload["memo"]

    normalized_answer = normalize_answer(expected_answer)
    normalized_memo = normalize_answer(memo)
    
    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError("AI question_text must be a non-empty string")
    if not isinstance(memo, str) or not memo.strip():
        raise ValueError("AI memo must be a non-empty string")
    if "todo" in question_text.lower() or "todo" in memo.lower():
        raise ValueError("AI response contains a placeholder")
    if normalized_answer not in normalized_memo:
        raise ValueError("AI memo must include the verified answer")
    if _contains_verified_answer(question_text, expected_answer, question_type):
        raise ValueError("AI question_text reveals the verified answer")
    if "equation" in visible_information and _display_equation(equation) not in question_text.replace("*", ""):
        raise ValueError("AI question_text omitted the visible equation")
    question_type_phrase = question_type.replace("_", "-")
    if question_type_phrase not in question_text.lower():
        raise ValueError("AI question_text does not match the requested type")
    for hidden_item in hidden_information:
        if hidden_item == "gradient" and "gradient:" in question_text.lower():
            raise ValueError("AI question_text reveals hidden gradient information")
    return AIQuestionText(question_text=question_text.strip(), memo=memo.strip())


def _prompts(
    *,
    grade: int,
    difficulty: str,
    question_type: str,
    equation: str,
    expected_answer: str,
    gradient: int | float,
    x_intercept: int | float | None,
    y_intercept: int | float,
    visible_information: list[str],
    hidden_information: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "You write South African school Mathematics questions. Write clear, concise, "
        "age-appropriate wording for the supplied grade. The supplied mathematics is "
        "verified and must not be changed. Return only the requested JSON object. "
        "The graph has no legend, title, or equation. Use visible information, never "
        "reveal hidden information, and do not introduce unrelated concepts. The memo "
        "must explain the method clearly and end with the verified answer exactly."
    )
    user_prompt = json.dumps(
        {
            "grade": grade,
            "difficulty": difficulty,
            "question_type": question_type,
            "equation": equation,
            "verified_answer": expected_answer,
            "gradient": gradient,
            "x_intercept": x_intercept,
            "y_intercept": y_intercept,
            "visible_information": visible_information,
            "hidden_information": hidden_information,
            "output_requirements": {
                "question_text": "One learner-facing question. Include f(x) and the equation if equation is visible.",
                "memo": "A concise worked method ending with the verified answer.",
            },
        },
        indent=2,
    )
    return system_prompt, user_prompt


def write_linear_question(
    *,
    grade: int,
    difficulty: str,
    question_type: str,
    equation: str,
    expected_answer: str,
    gradient: int | float,
    x_intercept: int | float | None,
    y_intercept: int | float,
    visible_information: list[str],
    hidden_information: list[str],
) -> AIQuestionText:
    """Generate and validate AI wording, retrying one invalid response."""

    prompts = _prompts(
        grade=grade,
        difficulty=difficulty,
        question_type=question_type,
        equation=equation,
        expected_answer=expected_answer,
        gradient=gradient,
        x_intercept=x_intercept,
        y_intercept=y_intercept,
        visible_information=visible_information,
        hidden_information=hidden_information,
    )
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = create_structured_response(
                system_prompt=prompts[0], user_prompt=prompts[1]
            )
            payload = json.loads(response.output_text)
            return _validate_response(
                payload,
                question_type=question_type,
                equation=equation,
                expected_answer=expected_answer,
                visible_information=visible_information,
                hidden_information=hidden_information,
            )
        except (Exception) as error:
            last_error = error
    raise ValueError("AI response failed validation after one retry") from last_error
