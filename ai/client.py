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


def create_structured_response(*, system_prompt: str, user_prompt: str) -> Any:
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
                "name": "linear_question_text",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "question_text": {"type": "string"},
                        "memo": {"type": "string"},
                    },
                    "required": ["question_text", "memo"],
                    "additionalProperties": False,
                },
            }
        },
    )
