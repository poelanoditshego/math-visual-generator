"""AI-assisted extraction of supported Linear skills from a lesson transcript."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ai.client import create_json_schema_response
from models.question_models import SUPPORTED_LINEAR_QUESTION_TYPES


@dataclass(frozen=True)
class TranscriptAnalysis:
    question_types: list[str]


_TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "question_types": {
            "type": "array",
            "items": {"type": "string", "enum": list(SUPPORTED_LINEAR_QUESTION_TYPES)},
            "minItems": 1,
        }
    },
    "required": ["question_types"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "Identify only the supported Linear question skills that are meaningfully taught "
    "in this lesson transcript. Do not solve problems, infer unstated related skills, "
    "or output mathematical values."
)


def _validate_payload(payload: object) -> TranscriptAnalysis:
    if not isinstance(payload, dict) or set(payload) != {"question_types"}:
        raise ValueError("Transcript analysis must contain exactly question_types.")
    question_types = payload["question_types"]
    if not isinstance(question_types, list):
        raise ValueError("Transcript question_types must be an array.")
    if not question_types:
        raise ValueError("No supported Linear question types could be identified from this transcript.")

    unique_types: list[str] = []
    for question_type in question_types:
        if question_type not in SUPPORTED_LINEAR_QUESTION_TYPES:
            raise ValueError(f"Unsupported Linear question type: {question_type!r}")
        if question_type not in unique_types:
            unique_types.append(question_type)
    if not unique_types:
        raise ValueError("No supported Linear question types could be identified from this transcript.")
    return TranscriptAnalysis(question_types=unique_types)


def analyze_transcript(transcript: str) -> TranscriptAnalysis:
    """Identify supported Linear skills using the existing lazy AI client."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Please provide a non-empty transcript.")
    try:
        response = create_json_schema_response(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=transcript.strip(),
            schema_name="linear_transcript_analysis",
            schema=_TRANSCRIPT_SCHEMA,
        )
        return _validate_payload(json.loads(response.output_text))
    except json.JSONDecodeError as error:
        raise ValueError("Transcript analysis returned an invalid response.") from error
    except ValueError:
        raise
    except (AttributeError, TypeError) as error:
        raise ValueError("Transcript analysis returned an invalid response.") from error
    except Exception as error:
        raise ValueError("Transcript analysis failed.") from error
