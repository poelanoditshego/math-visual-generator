"""AI-assisted extraction of supported Linear skills from an uploaded question image."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ai.client import create_vision_json_schema_response
from models.question_models import SUPPORTED_LINEAR_QUESTION_TYPES


@dataclass(frozen=True)
class ImageQuestionAnalysis:
    question_types: list[str]


_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "question_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": list(SUPPORTED_LINEAR_QUESTION_TYPES),
            },
            "minItems": 1,
        }
    },
    "required": ["question_types"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "Identify only the supported Linear question skills that are present "
    "in this Mathematics question image. Do not solve problems, infer unstated related skills, "
    "or output mathematical values."
)

_SUPPORTED_MEDIA_TYPES = {
    "image/png": "image/png",
    "png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
}


def _validate_payload(payload: object) -> ImageQuestionAnalysis:
    if not isinstance(payload, dict) or set(payload) != {"question_types"}:
        raise ValueError("Image analysis must contain exactly question_types.")
    question_types = payload["question_types"]
    if not isinstance(question_types, list):
        raise ValueError("Image question_types must be an array.")
    if not question_types:
        raise ValueError("No supported Linear question types could be identified from this image.")

    unique_types: list[str] = []
    for question_type in question_types:
        if question_type not in SUPPORTED_LINEAR_QUESTION_TYPES:
            raise ValueError(f"Unsupported Linear question type: {question_type!r}")
        if question_type not in unique_types:
            unique_types.append(question_type)

    if not unique_types:
        raise ValueError("No supported Linear question types could be identified from this image.")
    return ImageQuestionAnalysis(question_types=unique_types)


def analyze_image_question(
    image_bytes: bytes,
    media_type: str,
) -> ImageQuestionAnalysis:
    """Identify supported Linear skills from an uploaded question image."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Please provide a non-empty image file.")
    if not isinstance(media_type, str) or media_type.lower() not in _SUPPORTED_MEDIA_TYPES:
        raise ValueError("Unsupported image format. Please upload a PNG or JPEG image.")

    normalized_media_type = _SUPPORTED_MEDIA_TYPES[media_type.lower()]

    try:
        response = create_vision_json_schema_response(
            system_prompt=_SYSTEM_PROMPT,
            image_bytes=image_bytes,
            media_type=normalized_media_type,
            schema_name="linear_image_question_analysis",
            schema=_ANALYSIS_SCHEMA,
        )
        return _validate_payload(json.loads(response.output_text))
    except json.JSONDecodeError as error:
        raise ValueError(f"Image question analysis returned an invalid response: {error}") from error
    except ValueError:
        raise
    except (AttributeError, TypeError) as error:
        raise ValueError(f"Image question analysis returned an invalid response: {error}") from error
    except Exception as error:
        raise ValueError(f"Image question analysis failed: {error}") from error
