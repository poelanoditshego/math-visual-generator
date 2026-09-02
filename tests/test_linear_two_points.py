"""Focused coverage for equation-from-two-points Linear questions."""

from pathlib import Path
import random
import shutil

import pytest

from ai.question_writer import _validate_response
from ai.example_question_analyzer import _ANALYSIS_SCHEMA as EXAMPLE_SCHEMA
from ai.image_question_analyzer import _ANALYSIS_SCHEMA as IMAGE_SCHEMA
from ai.transcript_analyzer import _TRANSCRIPT_SCHEMA as TRANSCRIPT_SCHEMA
from models.question_models import LinearQuestionData, QuestionBlueprint
from questions.engine import generate_question_batch
from questions.linear import build_memo, generate_linear_question_batch
from questions.specs import get_question_spec


QUESTION_TYPE = "equation_from_two_points"


def _remove_batch(batch) -> None:
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_equation_from_two_points_is_registered_and_available_everywhere():
    spec = get_question_spec(QUESTION_TYPE)

    assert spec.answer_type == "equation"
    assert spec.requires_graph is True
    assert spec.fingerprint_fields == ("canonical_point_pair",)
    assert QUESTION_TYPE in EXAMPLE_SCHEMA["properties"]["question_type"]["enum"]
    assert QUESTION_TYPE in TRANSCRIPT_SCHEMA["properties"]["question_types"]["items"]["enum"]
    assert QUESTION_TYPE in IMAGE_SCHEMA["properties"]["question_types"]["items"]["enum"]


def test_generated_points_and_python_answer_are_mathematically_consistent():
    batch = generate_linear_question_batch(
        QuestionBlueprint(number_of_questions=6, question_types=[QUESTION_TYPE]),
        seed=1204,
    )
    try:
        for question in batch.questions:
            data = question.mathematical_data
            assert data.point_a is not None
            assert data.point_b is not None
            assert data.point_a != data.point_b

            x1, y1 = data.point_a
            x2, y2 = data.point_b
            assert x1 != x2
            assert isinstance(data.gradient, int)
            assert isinstance(data.y_intercept, int)
            assert y1 == data.gradient * x1 + data.y_intercept
            assert y2 == data.gradient * x2 + data.y_intercept
            assert (y2 - y1) / (x2 - x1) == data.gradient
            assert question.expected_answer == f"y = {data.equation.replace('*x', 'x')}"
            assert question.expected_answer not in question.question_text
            assert f"A({x1}; {y1})" in question.question_text
            assert f"B({x2}; {y2})" in question.question_text
            assert f"m = {data.gradient}" in question.memo
            assert f"c = {data.y_intercept}" in question.memo
            assert question.memo.endswith(f"Therefore, {question.expected_answer}.")
    finally:
        _remove_batch(batch)


def test_graph_reuses_linear_renderer_inputs_and_hides_solution_annotations():
    batch = generate_question_batch(
        QuestionBlueprint(number_of_questions=1, question_types=[QUESTION_TYPE]),
        seed=87,
    )
    try:
        question = batch.questions[0]
        request = question.graph_request
        display = request.display

        assert request.graph_type == "Linear"
        assert request.equation == question.mathematical_data.equation
        assert display.additional_x_values == [
            question.mathematical_data.point_a[0],
            question.mathematical_data.point_b[0],
        ]
        assert display.additional_point_labels == ["A", "B"]
        assert display.show_additional_point_labels is True
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False
        assert display.show_gradient is False
        assert display.show_gradient_triangle is False
        assert display.show_x_intercepts is False
        assert display.show_y_intercepts is False
        assert Path(question.graph_artifact.image_path).is_file()
        assert Path(question.graph_artifact.image_path).stat().st_size > 0
    finally:
        _remove_batch(batch)


def test_known_two_point_memo_uses_gradient_and_substitution_method():
    data = LinearQuestionData(
        equation="2*x + 1",
        gradient=2,
        y_intercept=1,
        x_intercept=-0.5,
        point_a=(1, 3),
        point_b=(4, 9),
    )

    answer, memo = build_memo(QUESTION_TYPE, data)

    assert answer == "y = 2x + 1"
    assert "m = (9 - 3) / (4 - 1)" in memo
    assert "m = 6 / 3" in memo
    assert "m = 2" in memo
    assert "3 = 2(1) + c" in memo
    assert "c = 1" in memo
    assert memo.endswith("Therefore, y = 2x + 1.")


def test_seed_reproduces_two_point_mathematics():
    blueprint = QuestionBlueprint(number_of_questions=3, question_types=[QUESTION_TYPE])
    first = generate_linear_question_batch(blueprint, seed=5001)
    second = generate_linear_question_batch(blueprint, seed=5001)
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


def test_canonical_point_pair_ignores_a_b_order():
    common = dict(equation="2*x + 1", gradient=2, y_intercept=1, x_intercept=-0.5)
    forward = LinearQuestionData(**common, point_a=(1, 3), point_b=(4, 9))
    reversed_points = LinearQuestionData(**common, point_a=(4, 9), point_b=(1, 3))

    assert forward.canonical_point_pair == reversed_points.canonical_point_pair


@pytest.mark.parametrize(
    ("difficulty", "gradient_limit", "intercept_limit", "coordinate_limit"),
    [("Easy", 2, 4, 8), ("Medium", 4, 6, 12), ("Hard", 5, 8, 16)],
)
def test_difficulty_keeps_integer_values_within_configured_limits(
    difficulty, gradient_limit, intercept_limit, coordinate_limit
):
    from questions.linear import _generate_two_point_data

    data = _generate_two_point_data(random.Random(44), difficulty)

    assert 0 < abs(data.gradient) <= gradient_limit
    assert abs(data.y_intercept) <= intercept_limit
    assert all(
        abs(coordinate) <= coordinate_limit
        for point in (data.point_a, data.point_b)
        for coordinate in point
    )


def test_ai_validation_requires_exact_points_and_rejects_hidden_equation():
    common = dict(
        question_type=QUESTION_TYPE,
        equation="2*x + 1",
        expected_answer="y = 2x + 1",
        visible_information=["point A", "point B"],
        hidden_information=["equation", "gradient", "y-intercept"],
        point_a=(1, 3),
        point_b=(4, 9),
    )
    valid = {
        "question_text": (
            "Points A(1; 3) and B(4; 9) lie on a line. "
            "Determine the equation of the line passing through them."
        ),
        "memo": "Use the two points. Therefore, y = 2x + 1.",
    }

    assert _validate_response(valid, **common).question_text == valid["question_text"]
    with pytest.raises(ValueError, match="altered or omitted"):
        _validate_response(
            {**valid, "question_text": valid["question_text"].replace("B(4; 9)", "B(5; 9)")},
            **common,
        )
    with pytest.raises(ValueError, match="reveals"):
        _validate_response(
            {**valid, "question_text": valid["question_text"] + " The equation is y = 2x + 1."},
            **common,
        )
