"""Focused coverage for equation-from-gradient-and-point questions."""

from pathlib import Path
import random
import shutil

import pytest

from ai.example_question_analyzer import _ANALYSIS_SCHEMA as EXAMPLE_SCHEMA
from ai.image_question_analyzer import _ANALYSIS_SCHEMA as IMAGE_SCHEMA
from ai.question_writer import _validate_response
from ai.transcript_analyzer import _TRANSCRIPT_SCHEMA as TRANSCRIPT_SCHEMA
from models.question_models import LinearQuestionData, QuestionBlueprint
from questions.engine import generate_question_batch
from questions.linear import build_memo, generate_linear_question_batch
from questions.specs import get_question_spec


QUESTION_TYPE = "equation_from_gradient_and_point"


def _remove_batch(batch) -> None:
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_gradient_and_point_type_is_registered_and_shared_with_analyzers():
    spec = get_question_spec(QUESTION_TYPE)

    assert spec.family == "linear"
    assert spec.answer_type == "equation"
    assert spec.requires_graph is True
    assert spec.fingerprint_fields == ("gradient", "point_a")
    assert QUESTION_TYPE in EXAMPLE_SCHEMA["properties"]["question_type"]["enum"]
    assert QUESTION_TYPE in TRANSCRIPT_SCHEMA["properties"]["question_types"]["items"]["enum"]
    assert QUESTION_TYPE in IMAGE_SCHEMA["properties"]["question_types"]["items"]["enum"]


def test_generated_math_answer_and_memo_are_authoritative():
    batch = generate_linear_question_batch(
        QuestionBlueprint(number_of_questions=6, question_types=[QUESTION_TYPE]),
        seed=1317,
    )
    try:
        for question in batch.questions:
            data = question.mathematical_data
            assert isinstance(data.gradient, int)
            assert data.gradient != 0
            assert isinstance(data.y_intercept, int)
            assert data.point_a is not None
            x_value, y_value = data.point_a
            assert isinstance(x_value, int)
            assert isinstance(y_value, int)
            assert y_value == data.gradient * x_value + data.y_intercept
            assert question.expected_answer == f"y = {data.equation.replace('*x', 'x')}"
            assert f"gradient of {data.gradient}" in question.question_text
            assert f"A({x_value}; {y_value})" in question.question_text
            assert question.expected_answer not in question.question_text
            assert f"c = {data.y_intercept}" in question.memo
            assert question.memo.endswith(f"Therefore, {question.expected_answer}.")
    finally:
        _remove_batch(batch)


def test_graph_reuses_linear_renderer_and_shows_only_point_a():
    batch = generate_question_batch(
        QuestionBlueprint(number_of_questions=1, question_types=[QUESTION_TYPE]),
        seed=419,
    )
    try:
        question = batch.questions[0]
        request = question.graph_request
        display = request.display

        assert request.graph_type == "Linear"
        assert request.equation == question.mathematical_data.equation
        assert display.additional_x_values == [question.mathematical_data.point_a[0]]
        assert display.additional_point_labels == ["A"]
        assert display.show_additional_point_labels is True
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False
        assert display.show_gradient is False
        assert display.show_gradient_triangle is False
        assert display.show_x_intercepts is False
        assert display.show_y_intercepts is False
        graph_path = Path(question.graph_artifact.image_path)
        assert graph_path.is_file()
        assert graph_path.stat().st_size > 0
    finally:
        _remove_batch(batch)


def test_known_gradient_and_point_memo_calculates_c():
    data = LinearQuestionData(
        equation="3*x + 1",
        gradient=3,
        y_intercept=1,
        x_intercept=-1 / 3,
        point_a=(2, 7),
    )

    answer, memo = build_memo(QUESTION_TYPE, data)

    assert answer == "y = 3x + 1"
    assert "m = 3" in memo
    assert "A(2; 7)" in memo
    assert "7 = 3(2) + c" in memo
    assert "7 = 6 + c" in memo
    assert "c = 1" in memo
    assert memo.endswith("Therefore, y = 3x + 1.")


def test_same_seed_reproduces_gradient_and_point_math():
    blueprint = QuestionBlueprint(number_of_questions=3, question_types=[QUESTION_TYPE])
    first = generate_linear_question_batch(blueprint, seed=773)
    second = generate_linear_question_batch(blueprint, seed=773)
    try:
        assert [question.mathematical_data for question in first.questions] == [
            question.mathematical_data for question in second.questions
        ]
        assert [question.expected_answer for question in first.questions] == [
            question.expected_answer for question in second.questions
        ]
    finally:
        _remove_batch(first)
        _remove_batch(second)


def test_fingerprint_treats_the_same_gradient_and_point_as_duplicate():
    spec = get_question_spec(QUESTION_TYPE)
    first = LinearQuestionData("3*x + 1", 3, 1, -1 / 3, point_a=(2, 7))
    second = LinearQuestionData("3*x + 1", 3, 1, -1 / 3, point_a=(2, 7))

    first_fingerprint = (QUESTION_TYPE, *(getattr(first, field) for field in spec.fingerprint_fields))
    second_fingerprint = (QUESTION_TYPE, *(getattr(second, field) for field in spec.fingerprint_fields))
    assert first_fingerprint == second_fingerprint


@pytest.mark.parametrize(
    ("difficulty", "gradient_limit", "intercept_limit", "coordinate_limit"),
    [("Easy", 2, 4, 8), ("Medium", 4, 6, 12), ("Hard", 5, 8, 16)],
)
def test_difficulty_limits_remain_integer_friendly(
    difficulty, gradient_limit, intercept_limit, coordinate_limit
):
    from questions.linear import _generate_gradient_point_data

    data = _generate_gradient_point_data(random.Random(91), difficulty)

    assert 0 < abs(data.gradient) <= gradient_limit
    assert abs(data.y_intercept) <= intercept_limit
    assert all(abs(coordinate) <= coordinate_limit for coordinate in data.point_a)


def test_ai_validation_preserves_visible_gradient_and_point_and_hides_answer():
    common = dict(
        question_type=QUESTION_TYPE,
        equation="3*x + 1",
        expected_answer="y = 3x + 1",
        visible_information=["gradient", "point A"],
        hidden_information=["equation", "y-intercept"],
        gradient=3,
        point_a=(2, 7),
    )
    valid = {
        "question_text": (
            "A straight line has a gradient of 3 and passes through A(2; 7). "
            "Determine the equation of the line."
        ),
        "memo": "Substitute the point into y = mx + c. Therefore, y = 3x + 1.",
    }

    assert _validate_response(valid, **common).question_text == valid["question_text"]
    with pytest.raises(ValueError, match="gradient"):
        _validate_response(
            {**valid, "question_text": valid["question_text"].replace("gradient of 3", "gradient of 4")},
            **common,
        )
    with pytest.raises(ValueError, match="point"):
        _validate_response(
            {**valid, "question_text": valid["question_text"].replace("A(2; 7)", "A(2; 8)")},
            **common,
        )
    with pytest.raises(ValueError, match="reveals"):
        _validate_response(
            {**valid, "question_text": valid["question_text"] + " It is y = 3x + 1."},
            **common,
        )
    with pytest.raises(ValueError, match="y-intercept"):
        _validate_response(
            {**valid, "question_text": valid["question_text"] + " The y-intercept is 1."},
            **common,
        )
