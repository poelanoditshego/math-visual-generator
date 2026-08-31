import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai.example_question_analyzer import analyze_example_question


def _response(payload: object) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps(payload))


def test_empty_example_is_rejected_without_an_api_call():
    with patch("ai.example_question_analyzer.create_json_schema_response") as request:
        with pytest.raises(ValueError, match="non-empty example"):
            analyze_example_question("   ")
    request.assert_not_called()


def test_supported_ai_result_is_accepted_with_strict_schema():
    with patch(
        "ai.example_question_analyzer.create_json_schema_response",
        return_value=_response({"question_type": "x_intercept"}),
    ) as request:
        result = analyze_example_question("Determine the x-intercept of f(x) = 2x - 6.")

    assert result.question_type == "x_intercept"
    assert request.call_args.kwargs["schema_name"] == "linear_example_question_analysis"
    assert request.call_args.kwargs["schema"]["required"] == ["question_type"]


def test_unsupported_ai_type_is_rejected():
    with patch(
        "ai.example_question_analyzer.create_json_schema_response",
        return_value=_response({"question_type": "quadratic"}),
    ):
        with pytest.raises(ValueError, match="Unsupported Linear question type"):
            analyze_example_question("Determine the turning point.")


def test_malformed_structured_response_is_rejected():
    with patch(
        "ai.example_question_analyzer.create_json_schema_response",
        return_value=SimpleNamespace(output_text='{"question_type": "gradient", "reason": "extra"}'),
    ):
        with pytest.raises(ValueError, match="exactly question_type"):
            analyze_example_question("Determine the gradient.")


def test_invalid_json_response_is_rejected():
    with patch(
        "ai.example_question_analyzer.create_json_schema_response",
        return_value=SimpleNamespace(output_text="not JSON"),
    ):
        with pytest.raises(ValueError, match="invalid response"):
            analyze_example_question("Determine the gradient.")


def test_import_does_not_create_an_openai_client():
    with patch("ai.client.OpenAI") as openai:
        import ai.example_question_analyzer as analyzer

        importlib.reload(analyzer)
    openai.assert_not_called()

