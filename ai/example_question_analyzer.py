"""AI-assisted classification for pasted Linear example questions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ai.client import create_json_schema_response
from models.question_models import SUPPORTED_LINEAR_QUESTION_TYPES


@dataclass(frozen=True)
class ExampleQuestionAnalysis:
    question_type: str


_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "question_type": {
            "type": "string",
            "enum": list(SUPPORTED_LINEAR_QUESTION_TYPES),
        }
    },
    "required": ["question_type"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "Classify the supplied school Mathematics example as exactly one supported "
    "Linear question type. Do not solve it, explain it, or infer numerical values."
)


def _validate_payload(payload: object) -> ExampleQuestionAnalysis:
    if not isinstance(payload, dict) or set(payload) != {"question_type"}:
        raise ValueError("Example analysis must contain exactly question_type.")
    question_type = payload["question_type"]
    if question_type not in SUPPORTED_LINEAR_QUESTION_TYPES:
        raise ValueError(f"Unsupported Linear question type: {question_type!r}")
    return ExampleQuestionAnalysis(question_type=question_type)


def analyze_example_question(example_question: str) -> ExampleQuestionAnalysis:
    """Classify a non-empty example question through the existing lazy AI client."""
    if not isinstance(example_question, str) or not example_question.strip():
        raise ValueError("Please provide a non-empty example question.")
    try:
        response = create_json_schema_response(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=example_question.strip(),
            schema_name="linear_example_question_analysis",
            schema=_ANALYSIS_SCHEMA,
        )
        return _validate_payload(json.loads(response.output_text))
    except json.JSONDecodeError as error:
        raise ValueError("Example question analysis returned an invalid response.") from error
    except ValueError:
        raise
    except (AttributeError, TypeError) as error:
        raise ValueError("Example question analysis returned an invalid response.") from error
    except Exception as error:
        raise ValueError("Example question analysis failed.") from error
