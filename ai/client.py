from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"


def get_client() -> OpenAI:
    """Create the SDK client lazily so importing the AI package makes no API call."""

    return OpenAI()


def get_model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def create_json_schema_response(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
) -> Any:
    """Create a lazy Responses API request using a strict JSON schema."""
    client = get_client()
    return client.responses.create(
        model=get_model(),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )


def create_structured_response(*, system_prompt: str, user_prompt: str) -> Any:
    """Create the existing strict response used by the question writer."""
    return create_json_schema_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="linear_question_text",
        schema={
            "type": "object",
            "properties": {
                "question_text": {"type": "string"},
                "memo": {"type": "string"},
            },
            "required": ["question_text", "memo"],
            "additionalProperties": False,
        },
    )


def create_vision_json_schema_response(
    *,
    system_prompt: str,
    user_prompt: str = "Analyze this Mathematics question image.",
    image_bytes: bytes,
    media_type: str,
    schema_name: str,
    schema: dict[str, Any],
) -> Any:
    """Create a lazy Responses API request with image input using a strict JSON schema."""
    import base64

    client = get_client()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    normalized_media = "image/jpeg" if media_type.lower() in ("image/jpeg", "image/jpg") else media_type.lower()
    image_url = f"data:{normalized_media};base64,{base64_data}"
    return client.responses.create(
        model=get_model(),
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )


