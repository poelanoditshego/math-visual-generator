from pathlib import Path
import shutil
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ai import example_question_analyzer, image_question_analyzer, transcript_analyzer
from ai.example_question_analyzer import ExampleQuestionAnalysis
from ai.image_question_analyzer import ImageQuestionAnalysis
from ai.question_writer import _validate_response
from ai.transcript_analyzer import TranscriptAnalysis
from generators.mixed import create_mixed_graph
from models.graph_settings import GraphSettings
from models.question_models import QuestionBlueprint
from questions.blueprints import (
    build_example_question_blueprint,
    build_image_question_blueprint,
    build_topic_question_blueprint,
    build_transcript_question_blueprint,
)
from questions.linear import generate_linear_question_batch
from questions.specs import get_question_spec


QUESTION_TYPE = "intersection_of_two_lines"


def _blueprint(count: int = 1) -> QuestionBlueprint:
    return QuestionBlueprint(number_of_questions=count, question_types=[QUESTION_TYPE])


def _remove_batch(batch) -> None:
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def _calculated_intersection(data) -> tuple[float, float]:
    x_value = (
        data.second_y_intercept - data.y_intercept
    ) / (data.gradient - data.second_gradient)
    return x_value, data.gradient * x_value + data.y_intercept


def test_intersection_spec_hides_answer_but_keeps_graph_readable():
    spec = get_question_spec(QUESTION_TYPE)
    display = spec.build_display_settings()
    assert spec.answer_type == "coordinate"
    assert display.show_intersection_points is False
    assert display.show_point_labels is False
    assert display.show_grid is True
    assert display.show_tick_labels is True
    assert display.graph_curve_label_style == "Function name only"


def test_wrapper_generates_exact_two_line_question_through_mixed_graph_path():
    batch = generate_linear_question_batch(_blueprint(), seed=314)
    try:
        question = batch.questions[0]
        data = question.mathematical_data
        calculated_x, calculated_y = _calculated_intersection(data)

        assert data.second_equation is not None
        assert data.gradient != data.second_gradient
        assert data.intersection_point == (calculated_x, calculated_y)
        assert all(float(value).is_integer() for value in data.intersection_point)
        assert question.expected_answer == f"({int(calculated_x)}; {int(calculated_y)})"
        assert question.graph_request.graph_type == "Mixed"
        assert question.graph_request.equation is None
        assert question.graph_request.equations == [data.equation, data.second_equation]
        assert question.graph_request.display.show_intersection_points is False
        assert question.expected_answer not in question.question_text
        assert Path(question.graph_artifact.image_path).is_file()
        assert question.graph_artifact.graph_type == "Mixed"

        graph_range = question.graph_request.graph_range
        assert graph_range.x_min < calculated_x < graph_range.x_max
        assert graph_range.y_min < calculated_y < graph_range.y_max
        assert "f(x) = g(x)" in question.memo
        assert data.equation.replace("*x", "x") in question.memo
        assert data.second_equation.replace("*x", "x") in question.memo
        assert f"x = {int(calculated_x)}" in question.memo
        assert question.memo.endswith(f"{question.expected_answer}.")
    finally:
        _remove_batch(batch)


def test_same_seed_reproduces_question_and_batch_fingerprints_are_unique():
    first = generate_linear_question_batch(_blueprint(5), seed=902)
    second = generate_linear_question_batch(_blueprint(5), seed=902)
    try:
        assert [q.mathematical_data for q in first.questions] == [
            q.mathematical_data for q in second.questions
        ]
        fingerprints = [
            (
                q.mathematical_data.canonical_line_pair,
                q.mathematical_data.intersection_point,
            )
            for q in first.questions
        ]
        assert len(fingerprints) == len(set(fingerprints))
    finally:
        _remove_batch(first)
        _remove_batch(second)


@patch("generators.mixed.plt.close")
@patch("generators.mixed.plt.savefig")
def test_existing_mixed_renderer_draws_no_answer_marker_when_disabled(savefig, close):
    settings = GraphSettings(
        x_min=-5,
        x_max=5,
        y_min=-10,
        y_max=10,
        show_x_intercepts=False,
        show_y_intercepts=False,
        show_intersection_points=False,
        show_point_labels=False,
        graph_curve_label_style="Function name only",
        output_name="hidden-linear-intersection.png",
    )
    create_mixed_graph(["2*x + 1", "-x + 4"], settings)
    try:
        ax = plt.gca()
        graph_lines = [line for line in ax.lines if line.get_linewidth() == 2]
        assert len(graph_lines) == 2
        assert not ax.collections
        labels = {text.get_text() for text in ax.texts}
        assert "$f$" in labels
        assert "$g$" in labels
        assert not any("1" in label and "3" in label for label in labels)
    finally:
        plt.clf()


def test_ai_validation_rejects_intersection_coordinate_leakage():
    with pytest.raises(ValueError, match="reveals the verified answer"):
        _validate_response(
            {
                "question_text": "Determine the point of intersection (1; 3).",
                "memo": "Solve f(x) = g(x). Therefore the answer is (1; 3).",
            },
            question_type=QUESTION_TYPE,
            equation="2*x + 1",
            expected_answer="(1; 3)",
            visible_information=["graphs f and g"],
            hidden_information=["intersection coordinates"],
        )


def test_all_input_modes_accept_intersection_blueprints():
    common = {"grade": 9, "difficulty": "Medium", "number_of_questions": 1}
    blueprints = [
        build_topic_question_blueprint([QUESTION_TYPE], **common),
        build_example_question_blueprint(ExampleQuestionAnalysis(QUESTION_TYPE), **common),
        build_transcript_question_blueprint(TranscriptAnalysis([QUESTION_TYPE]), **common),
        build_image_question_blueprint(ImageQuestionAnalysis([QUESTION_TYPE]), **common),
    ]
    for blueprint in blueprints:
        blueprint.validate()
        assert blueprint.question_types == [QUESTION_TYPE]


def test_all_analyzer_schemas_offer_the_new_type():
    assert QUESTION_TYPE in example_question_analyzer._ANALYSIS_SCHEMA["properties"][
        "question_type"
    ]["enum"]
    assert QUESTION_TYPE in transcript_analyzer._TRANSCRIPT_SCHEMA["properties"][
        "question_types"
    ]["items"]["enum"]
    assert QUESTION_TYPE in image_question_analyzer._ANALYSIS_SCHEMA["properties"][
        "question_types"
    ]["items"]["enum"]
