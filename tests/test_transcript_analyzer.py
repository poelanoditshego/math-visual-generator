import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai.transcript_analyzer import analyze_transcript


def _response(payload: object) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps(payload))


def test_empty_transcript_is_rejected_without_an_api_call():
    with patch("ai.transcript_analyzer.create_json_schema_response") as request:
        with pytest.raises(ValueError, match="non-empty transcript"):
            analyze_transcript(" \n ")
    request.assert_not_called()


def test_supported_types_are_accepted_and_deduplicated_in_order():
    with patch(
        "ai.transcript_analyzer.create_json_schema_response",
        return_value=_response(
            {"question_types": ["gradient", "x_intercept", "gradient", "y_intercept"]}
        ),
    ) as request:
        result = analyze_transcript("We discuss gradients and both intercepts.")

    assert result.question_types == ["gradient", "x_intercept", "y_intercept"]
    schema = request.call_args.kwargs["schema"]
    assert request.call_args.kwargs["schema_name"] == "linear_transcript_analysis"
    assert schema["required"] == ["question_types"]
    assert schema["properties"]["question_types"]["minItems"] == 1


def test_empty_detected_type_list_is_rejected():
    with patch(
        "ai.transcript_analyzer.create_json_schema_response",
        return_value=_response({"question_types": []}),
    ):
        with pytest.raises(ValueError, match="No supported Linear"):
            analyze_transcript("A lesson with no supported skill.")


def test_unsupported_type_is_rejected():
    with patch(
        "ai.transcript_analyzer.create_json_schema_response",
        return_value=_response({"question_types": ["quadratic"]}),
    ):
        with pytest.raises(ValueError, match="Unsupported Linear question type"):
            analyze_transcript("Find the turning point.")


def test_non_array_type_list_is_rejected():
    with patch(
        "ai.transcript_analyzer.create_json_schema_response",
        return_value=_response({"question_types": "gradient"}),
    ):
        with pytest.raises(ValueError, match="must be an array"):
            analyze_transcript("Determine the gradient.")


def test_malformed_structured_response_is_rejected():
    with patch(
        "ai.transcript_analyzer.create_json_schema_response",
        return_value=_response({"question_types": ["gradient"], "reason": "extra"}),
    ):
        with pytest.raises(ValueError, match="exactly question_types"):
            analyze_transcript("Determine the gradient.")


def test_invalid_json_response_is_rejected():
    with patch(
        "ai.transcript_analyzer.create_json_schema_response",
        return_value=SimpleNamespace(output_text="not JSON"),
    ):
        with pytest.raises(ValueError, match="invalid response"):
            analyze_transcript("Determine the gradient.")


def test_import_does_not_create_an_openai_client():
    with patch("ai.client.OpenAI") as openai:
        import ai.transcript_analyzer as analyzer

        importlib.reload(analyzer)
    openai.assert_not_called()
