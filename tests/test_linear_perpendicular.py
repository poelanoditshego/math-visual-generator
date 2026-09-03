"""Focused tests for perpendicular_lines Straight Line questions."""

from pathlib import Path
import random
import shutil

import pytest

from ai.example_question_analyzer import _ANALYSIS_SCHEMA as EXAMPLE_SCHEMA
from ai.image_question_analyzer import _ANALYSIS_SCHEMA as IMAGE_SCHEMA
from ai.question_writer import _validate_response
from ai.transcript_analyzer import _TRANSCRIPT_SCHEMA as TRANSCRIPT_SCHEMA
from generators.api import graph_request_to_settings
from models.question_models import LinearQuestionData, QuestionBlueprint
from questions.engine import generate_question_batch
from questions.linear import (
    _generate_perpendicular_lines_data,
    build_memo,
    generate_linear_question_batch,
)
from questions.specs import get_question_spec


QUESTION_TYPE = "perpendicular_lines"


def _remove_batch(batch) -> None:
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_registered_and_available_in_analyzers():
    spec = get_question_spec(QUESTION_TYPE)
    assert spec.family == "linear"
    assert spec.answer_type == "equation"
    assert spec.requires_graph is True
    assert spec.fingerprint_fields == ("equation", "second_equation", "point_a")
    assert "gradient of g" in spec.ai_hidden_information
    assert QUESTION_TYPE in EXAMPLE_SCHEMA["properties"]["question_type"]["enum"]
    assert QUESTION_TYPE in TRANSCRIPT_SCHEMA["properties"]["question_types"]["items"]["enum"]
    assert QUESTION_TYPE in IMAGE_SCHEMA["properties"]["question_types"]["items"]["enum"]


def test_generated_math_answer_memo_and_question_are_consistent():
    batch = generate_linear_question_batch(
        QuestionBlueprint(number_of_questions=6, question_types=[QUESTION_TYPE]),
        seed=2026,
    )
    try:
        for question in batch.questions:
            data = question.mathematical_data
            assert data.second_equation is not None
            assert data.second_gradient is not None
            assert data.second_y_intercept is not None
            assert data.point_a is not None
            assert data.gradient != 0
            assert data.gradient * data.second_gradient == -1
            assert (data.gradient, data.second_gradient) in {(1, -1), (-1, 1)}
            assert data.equation != data.second_equation

            x_a, y_a = data.point_a
            assert y_a == data.second_gradient * x_a + data.second_y_intercept
            g_display = data.second_equation.replace("*x", "x")
            assert question.expected_answer == f"g(x) = {g_display}"
            assert f"f(x) = {data.equation.replace('*x', 'x')}" in question.question_text
            assert f"A({x_a}; {y_a})" in question.question_text
            assert "perpendicular" in question.question_text.lower()
            assert question.expected_answer not in question.question_text
            assert "m_f × m_g = -1" in question.memo
            assert f"m_g = {data.second_gradient}" in question.memo
            assert f"c = {data.second_y_intercept}" in question.memo
            assert question.memo.endswith(f"Therefore, {question.expected_answer}.")
    finally:
        _remove_batch(batch)


def test_graph_reuses_mixed_renderer_and_places_only_a_on_g():
    batch = generate_question_batch(
        QuestionBlueprint(number_of_questions=1, question_types=[QUESTION_TYPE]), seed=101
    )
    try:
        question = batch.questions[0]
        request, display = question.graph_request, question.graph_request.display
        assert request.graph_type == "Mixed"
        assert request.equations == [
            question.mathematical_data.equation,
            question.mathematical_data.second_equation,
        ]
        assert display.additional_x_values == [question.mathematical_data.point_a[0]]
        assert display.additional_point_labels == ["A"]
        assert display.additional_point_function_indices == [1]
        assert graph_request_to_settings(request).additional_point_function_indices == [1]
        assert display.show_additional_point_labels is True
        assert display.show_point_labels is False
        assert display.show_title is False
        assert display.show_legend is False
        assert display.show_equation is False
        assert display.show_gradient is False
        assert display.show_gradient_triangle is False
        assert display.show_x_intercepts is False
        assert display.show_y_intercepts is False
        assert display.show_intersection_points is False
        graph_path = Path(question.graph_artifact.image_path)
        assert graph_path.is_file() and graph_path.stat().st_size > 0

        data = question.mathematical_data
        crossing_x = (data.second_y_intercept - data.y_intercept) / (
            data.gradient - data.second_gradient
        )
        crossing_y = data.gradient * crossing_x + data.y_intercept
        graph_range = request.graph_range
        assert graph_range.x_min < crossing_x < graph_range.x_max
        assert graph_range.y_min < crossing_y < graph_range.y_max
    finally:
        _remove_batch(batch)


def test_known_memo_derives_gradient_and_intercept():
    data = LinearQuestionData(
        equation="x + 1", gradient=1, y_intercept=1, x_intercept=-1,
        second_equation="-x + 7", second_gradient=-1, second_y_intercept=7,
        point_a=(4, 3),
    )
    answer, memo = build_memo(QUESTION_TYPE, data)
    assert answer == "g(x) = -x + 7"
    assert "m_f = 1" in memo
    assert "m_f × m_g = -1" in memo
    assert "1m_g = -1" in memo
    assert "m_g = -1" in memo
    assert "3 = -1(4) + c" in memo
    assert "3 = -4 + c" in memo
    assert "c = 7" in memo


def test_same_seed_reproduces_math_and_fingerprint_preserves_roles():
    blueprint = QuestionBlueprint(number_of_questions=3, question_types=[QUESTION_TYPE])
    first = generate_linear_question_batch(blueprint, seed=888)
    second = generate_linear_question_batch(blueprint, seed=888)
    try:
        assert [q.mathematical_data for q in first.questions] == [
            q.mathematical_data for q in second.questions
        ]
        spec = get_question_spec(QUESTION_TYPE)
        data = first.questions[0].mathematical_data
        swapped = LinearQuestionData(
            equation=data.second_equation, gradient=data.second_gradient,
            y_intercept=data.second_y_intercept,
            x_intercept=-data.second_y_intercept / data.second_gradient,
            second_equation=data.equation, second_gradient=data.gradient,
            second_y_intercept=data.y_intercept, point_a=data.point_a,
        )
        fp = (QUESTION_TYPE, *(getattr(data, f) for f in spec.fingerprint_fields))
        swapped_fp = (QUESTION_TYPE, *(getattr(swapped, f) for f in spec.fingerprint_fields))
        assert fp != swapped_fp
    finally:
        _remove_batch(first)
        _remove_batch(second)


@pytest.mark.parametrize(
    ("difficulty", "intercept_limit", "coordinate_limit"),
    [("Easy", 4, 8), ("Medium", 6, 12), ("Hard", 8, 16)],
)
def test_difficulty_limits_and_exact_integer_first_version(
    difficulty, intercept_limit, coordinate_limit
):
    data = _generate_perpendicular_lines_data(random.Random(42), difficulty)
    assert abs(data.gradient) == abs(data.second_gradient) == 1
    assert data.gradient * data.second_gradient == -1
    assert abs(data.y_intercept) <= intercept_limit
    assert abs(data.second_y_intercept) <= intercept_limit
    assert all(abs(coordinate) <= coordinate_limit for coordinate in data.point_a)


def test_ai_validation_preserves_inputs_and_rejects_solution_leakage():
    common = dict(
        question_type=QUESTION_TYPE, equation="x + 1",
        expected_answer="g(x) = -x + 7",
        visible_information=["reference equation", "point A", "perpendicular relationship"],
        hidden_information=["equation of g", "y-intercept of g", "gradient of g"],
        gradient=1, second_gradient=-1, point_a=(4, 3),
    )
    text = (
        "A straight line g is perpendicular to f(x) = x + 1 and passes through "
        "A(4; 3). Determine the equation of g."
    )
    valid = {"question_text": text, "memo": "Work gives g(x) = -x + 7"}
    assert _validate_response(valid, **common).question_text == text

    invalid = [
        (text.replace("f(x) = x + 1", "f"), "reference line equation"),
        (text.replace("A(4; 3)", "A(4; 4)"), "altered or omitted"),
        (text.replace("perpendicular", "parallel"), "perpendicular"),
        (text + " Thus g(x) = -x + 7.", "verified answer"),
        (text + " The gradient of g is -1.", "hidden gradient"),
        (text + " The y-intercept is 7.", "hidden y-intercept"),
    ]
    for question_text, message in invalid:
        with pytest.raises(ValueError, match=message):
            _validate_response({**valid, "question_text": question_text}, **common)


def test_existing_parallel_lines_still_generates():
    batch = generate_linear_question_batch(
        QuestionBlueprint(number_of_questions=1, question_types=["parallel_lines"]), seed=7
    )
    try:
        data = batch.questions[0].mathematical_data
        assert data.gradient == data.second_gradient
    finally:
        _remove_batch(batch)
