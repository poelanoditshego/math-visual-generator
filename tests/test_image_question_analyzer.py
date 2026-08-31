import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai.image_question_analyzer import ImageQuestionAnalysis, analyze_image_question
from models.question_models import QuestionBlueprint


def _response(payload: object) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps(payload))


def test_empty_image_is_rejected_without_an_api_call():
    with patch("ai.image_question_analyzer.create_vision_json_schema_response") as request:
        with pytest.raises(ValueError, match="non-empty image file"):
            analyze_image_question(b"", "image/png")
    request.assert_not_called()


def test_unsupported_media_type_is_rejected_without_an_api_call():
    with patch("ai.image_question_analyzer.create_vision_json_schema_response") as request:
        with pytest.raises(ValueError, match="Unsupported image format"):
            analyze_image_question(b"fake_image_bytes", "application/pdf")
    request.assert_not_called()


def test_single_detected_type_accepted():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=_response({"question_types": ["x_intercept"]}),
    ) as request:
        result = analyze_image_question(b"png_data", "image/png")

    assert isinstance(result, ImageQuestionAnalysis)
    assert result.question_types == ["x_intercept"]
    assert request.call_args.kwargs["schema_name"] == "linear_image_question_analysis"
    assert request.call_args.kwargs["media_type"] == "image/png"


def test_multiple_detected_types_accepted():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=_response({
            "question_types": ["gradient", "increasing_or_decreasing", "x_intercept", "y_intercept"]
        }),
    ):
        result = analyze_image_question(b"jpeg_data", "image/jpeg")

    assert result.question_types == ["gradient", "increasing_or_decreasing", "x_intercept", "y_intercept"]


def test_duplicate_types_deduplicated_in_order():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=_response({"question_types": ["gradient", "x_intercept", "gradient", "y_intercept"]}),
    ):
        result = analyze_image_question(b"png_data", "image/png")

    assert result.question_types == ["gradient", "x_intercept", "y_intercept"]


def test_unsupported_type_is_rejected():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=_response({"question_types": ["quadratic"]}),
    ):
        with pytest.raises(ValueError, match="Unsupported Linear question type"):
            analyze_image_question(b"png_data", "image/png")


def test_empty_type_list_is_rejected():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=_response({"question_types": []}),
    ):
        with pytest.raises(ValueError, match="No supported Linear question types"):
            analyze_image_question(b"png_data", "image/png")


def test_malformed_structured_response_is_rejected():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=_response({"question_types": ["gradient"], "extra": "field"}),
    ):
        with pytest.raises(ValueError, match="exactly question_types"):
            analyze_image_question(b"png_data", "image/png")


def test_invalid_json_response_is_rejected():
    with patch(
        "ai.image_question_analyzer.create_vision_json_schema_response",
        return_value=SimpleNamespace(output_text="not JSON"),
    ):
        with pytest.raises(ValueError, match="invalid response"):
            analyze_image_question(b"png_data", "image/png")


def test_import_does_not_create_an_openai_client():
    with patch("ai.client.OpenAI") as openai:
        import ai.image_question_analyzer as analyzer

        importlib.reload(analyzer)
    openai.assert_not_called()


def test_blueprint_accepts_multiple_image_question_types():
    analysis = ImageQuestionAnalysis(
        question_types=["gradient", "increasing_or_decreasing", "x_intercept", "y_intercept"]
    )
    blueprint = QuestionBlueprint(
        grade=9,
        difficulty="Medium",
        number_of_questions=8,
        question_types=analysis.question_types,
    )
    blueprint.validate()
    assert blueprint.grade == 9
    assert blueprint.difficulty == "Medium"
    assert blueprint.number_of_questions == 8
    assert blueprint.question_types == ["gradient", "increasing_or_decreasing", "x_intercept", "y_intercept"]
