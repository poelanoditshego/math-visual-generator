"""Focused test suite for parallel_lines Straight Line questions."""

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


QUESTION_TYPE = "parallel_lines"


def _remove_batch(batch) -> None:
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_parallel_lines_registered_and_available_in_analyzers():
    spec = get_question_spec(QUESTION_TYPE)

    assert spec.family == "linear"
    assert spec.answer_type == "equation"
    assert spec.requires_graph is True
    assert spec.fingerprint_fields == ("equation", "second_equation", "point_a")
    assert QUESTION_TYPE in EXAMPLE_SCHEMA["properties"]["question_type"]["enum"]
    assert QUESTION_TYPE in TRANSCRIPT_SCHEMA["properties"]["question_types"]["items"]["enum"]
    assert QUESTION_TYPE in IMAGE_SCHEMA["properties"]["question_types"]["items"]["enum"]


def test_generated_math_and_python_answer_are_consistent():
    batch = generate_linear_question_batch(
        QuestionBlueprint(number_of_questions=6, question_types=[QUESTION_TYPE]),
        seed=2026,
    )
    try:
        for question in batch.questions:
            data: LinearQuestionData = question.mathematical_data
            assert data.gradient is not None
            assert data.second_gradient is not None
            assert data.y_intercept is not None
            assert data.second_y_intercept is not None
            assert data.point_a is not None
            assert data.second_equation is not None

            # 1. Gradients are equal
            assert data.gradient == data.second_gradient
            assert data.gradient != 0

            # 2. y-intercepts are different
            assert data.y_intercept != data.second_y_intercept

            # 3. Point A lies on target line g
            x_A, y_A = data.point_a
            assert y_A == data.second_gradient * x_A + data.second_y_intercept

            # 4. Point A does NOT lie on reference line f (since c_f != c_g)
            assert y_A != data.gradient * x_A + data.y_intercept

            # 5. Authoritative answer format is g(x) = ...
            g_display = data.second_equation.replace("*x", "x")
            assert question.expected_answer == f"g(x) = {g_display}"

            # 6. Reference equation f is visible in question text, target equation g is not
            f_display = data.equation.replace("*x", "x")
            assert f"f(x) = {f_display}" in question.question_text
            assert question.expected_answer not in question.question_text
            assert f"A({x_A}; {y_A})" in question.question_text

            # 7. Memo explains equal gradients and concludes with g(x) = ...
            assert "Parallel lines have equal gradients." in question.memo
            assert f"m = {data.gradient}" in question.memo
            assert f"c = {data.second_y_intercept}" in question.memo
            assert question.memo.endswith(f"Therefore, {question.expected_answer}.")
    finally:
        _remove_batch(batch)


def test_graph_uses_mixed_renderer_and_configures_point_a_on_g_only():
    batch = generate_question_batch(
        QuestionBlueprint(number_of_questions=1, question_types=[QUESTION_TYPE]),
        seed=101,
    )
    try:
        question = batch.questions[0]
        request = question.graph_request
        display = request.display

        assert request.graph_type == "Mixed"
        assert request.equations == [
            question.mathematical_data.equation,
            question.mathematical_data.second_equation,
        ]
        assert display.additional_x_values == [question.mathematical_data.point_a[0]]
        assert display.additional_point_labels == ["A"]
        assert display.additional_point_function_indices == [1]
        assert display.show_additional_point_labels is True
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False
        assert display.show_gradient is False
        assert display.show_gradient_triangle is False
        assert display.show_x_intercepts is False
        assert display.show_y_intercepts is False
        assert display.show_intersection_points is False

        graph_path = Path(question.graph_artifact.image_path)
        assert graph_path.is_file()
        assert graph_path.stat().st_size > 0
    finally:
        _remove_batch(batch)


def test_lines_never_intersect():
    batch = generate_linear_question_batch(
        QuestionBlueprint(number_of_questions=5, question_types=[QUESTION_TYPE]),
        seed=555,
    )
    try:
        for question in batch.questions:
            data = question.mathematical_data
            m_f, c_f = data.gradient, data.y_intercept
            m_g, c_g = data.second_gradient, data.second_y_intercept
            # Parallel distinct lines: m_f == m_g and c_f != c_g => no solution to m_f*x + c_f == m_g*x + c_g
            assert m_f == m_g
            assert c_f != c_g
    finally:
        _remove_batch(batch)


def test_known_parallel_lines_memo_solves_for_cg():
    data = LinearQuestionData(
        equation="2*x + 1",
        gradient=2,
        y_intercept=1,
        x_intercept=-0.5,
        second_equation="2*x + 4",
        second_gradient=2,
        second_y_intercept=4,
        point_a=(3, 10),
    )

    answer, memo = build_memo(QUESTION_TYPE, data)

    assert answer == "g(x) = 2x + 4"
    assert "Parallel lines have equal gradients." in memo
    assert "The gradient of f is m = 2." in memo
    assert "Therefore, the gradient of g is also m = 2." in memo
    assert "A(3; 10)" in memo
    assert "10 = 2(3) + c" in memo
    assert "10 = 6 + c" in memo
    assert "c = 4" in memo
    assert memo.endswith("Therefore, g(x) = 2x + 4.")


def test_seed_reproduces_parallel_lines_math():
    blueprint = QuestionBlueprint(number_of_questions=3, question_types=[QUESTION_TYPE])
    first = generate_linear_question_batch(blueprint, seed=888)
    second = generate_linear_question_batch(blueprint, seed=888)
    try:
        assert [q.mathematical_data for q in first.questions] == [
            q.mathematical_data for q in second.questions
        ]
        assert [q.expected_answer for q in first.questions] == [
            q.expected_answer for q in second.questions
        ]
    finally:
        _remove_batch(first)
        _remove_batch(second)


def test_fingerprint_preserves_line_order():
    spec = get_question_spec(QUESTION_TYPE)
    data1 = LinearQuestionData(
        equation="2*x + 1", gradient=2, y_intercept=1, x_intercept=-0.5,
        second_equation="2*x + 4", second_gradient=2, second_y_intercept=4,
        point_a=(3, 10),
    )
    # Swapped reference and target line
    data2 = LinearQuestionData(
        equation="2*x + 4", gradient=2, y_intercept=4, x_intercept=-2.0,
        second_equation="2*x + 1", second_gradient=2, second_y_intercept=1,
        point_a=(3, 10),
    )

    fp1 = (QUESTION_TYPE, *(getattr(data1, field) for field in spec.fingerprint_fields))
    fp2 = (QUESTION_TYPE, *(getattr(data2, field) for field in spec.fingerprint_fields))
    assert fp1 != fp2


@pytest.mark.parametrize(
    ("difficulty", "gradient_limit", "intercept_limit", "coordinate_limit"),
    [("Easy", 2, 4, 8), ("Medium", 4, 6, 12), ("Hard", 5, 8, 16)],
)
def test_difficulty_limits_parallel_lines(
    difficulty, gradient_limit, intercept_limit, coordinate_limit
):
    from questions.linear import _generate_parallel_lines_data

    data = _generate_parallel_lines_data(random.Random(42), difficulty)

    assert 0 < abs(data.gradient) <= gradient_limit
    assert abs(data.y_intercept) <= intercept_limit
    assert abs(data.second_y_intercept) <= intercept_limit
    assert all(abs(coord) <= coordinate_limit for coord in data.point_a)


def test_ai_validation_enforces_reference_equation_point_a_and_parallel_wording():
    common = dict(
        question_type=QUESTION_TYPE,
        equation="2*x + 1",
        expected_answer="g(x) = 2x + 4",
        visible_information=["reference equation", "point A", "parallel relationship"],
        hidden_information=["equation of g", "y-intercept of g"],
        gradient=2,
        point_a=(3, 10),
    )
    valid = {
        "question_text": (
            "A straight line g is parallel to f(x) = 2x + 1 and passes through "
            "the point A(3; 10). Determine the equation of g."
        ),
        "memo": "Parallel lines have equal gradients m = 2. Substitute A(3; 10): 10 = 2(3) + c, c = 4. Therefore, g(x) = 2x + 4.",
    }

    assert _validate_response(valid, **common).question_text == valid["question_text"]

    # Omitting reference equation
    with pytest.raises(ValueError, match="reference line equation"):
        _validate_response(
            {**valid, "question_text": "A line g is parallel to f and passes through A(3; 10). Determine equation of g."},
            **common,
        )

    # Omitting point A
    with pytest.raises(ValueError, match="altered or omitted"):
        _validate_response(
            {**valid, "question_text": valid["question_text"].replace("A(3; 10)", "A(3; 11)")},
            **common,
        )

    # Omitting parallel wording
    with pytest.raises(ValueError, match="parallel"):
        _validate_response(
            {**valid, "question_text": "A line g intersects f(x) = 2x + 1 at A(3; 10). Determine equation of g."},
            **common,
        )

    # Revealing target equation g(x) = 2x + 4
    with pytest.raises(ValueError, match="reveals the verified answer"):
        _validate_response(
            {**valid, "question_text": valid["question_text"] + " Note g(x) = 2x + 4."},
            **common,
        )
