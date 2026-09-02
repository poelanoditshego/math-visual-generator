from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from ai.example_question_analyzer import ExampleQuestionAnalysis
from ai.image_question_analyzer import ImageQuestionAnalysis
from ai.transcript_analyzer import TranscriptAnalysis
from questions.blueprints import (
    build_example_question_blueprint,
    build_image_question_blueprint,
    build_topic_question_blueprint,
    build_transcript_question_blueprint,
)
from models.question_models import QuestionBlueprint, SUPPORTED_LINEAR_QUESTION_TYPES
from questions.engine import generate_question_batch, get_family_generator
from questions.linear import generate_linear_question_batch
from questions.specs import LINEAR_QUESTION_SPECS, get_question_spec


def _remove_batch(batch) -> None:
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_every_supported_linear_type_has_one_registered_specification():
    assert tuple(LINEAR_QUESTION_SPECS) == SUPPORTED_LINEAR_QUESTION_TYPES
    assert all(spec.family == "linear" for spec in LINEAR_QUESTION_SPECS.values())


def test_registry_rejects_unknown_types():
    with pytest.raises(ValueError, match="Unsupported linear question type"):
        get_question_spec("not_registered", family="linear")


def test_generic_engine_dispatches_to_linear_and_wrapper_remains_compatible():
    blueprint = QuestionBlueprint(number_of_questions=1, question_types=["gradient"])
    generic = generate_question_batch(blueprint, seed=71)
    wrapped = generate_linear_question_batch(blueprint, seed=71)
    try:
        assert get_family_generator("linear").family == "linear"
        assert generic.questions[0].mathematical_data == wrapped.questions[0].mathematical_data
        assert generic.questions[0].expected_answer == wrapped.questions[0].expected_answer
    finally:
        _remove_batch(generic)
        _remove_batch(wrapped)


def test_generic_engine_is_deterministic_for_the_same_seed():
    blueprint = QuestionBlueprint(number_of_questions=2)
    first = generate_question_batch(blueprint, seed=702)
    second = generate_question_batch(blueprint, seed=702)
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


@pytest.mark.parametrize("question_type", SUPPORTED_LINEAR_QUESTION_TYPES)
def test_engine_generates_every_linear_type_with_python_answer(question_type):
    blueprint = QuestionBlueprint(number_of_questions=1, question_types=[question_type])
    with patch("questions.linear.write_linear_question") as writer:
        batch = generate_question_batch(blueprint, seed=19, use_ai=False)
    try:
        assert batch.questions[0].expected_answer
        writer.assert_not_called()
    finally:
        _remove_batch(batch)


def test_all_input_modes_build_the_same_valid_blueprint_shape():
    common = {"grade": 9, "difficulty": "Medium", "number_of_questions": 2}
    blueprints = [
        build_topic_question_blueprint(["gradient"], **common),
        build_example_question_blueprint(ExampleQuestionAnalysis("gradient"), **common),
        build_transcript_question_blueprint(TranscriptAnalysis(["gradient"]), **common),
        build_image_question_blueprint(ImageQuestionAnalysis(["gradient"]), **common),
    ]
    for blueprint in blueprints:
        blueprint.validate()
        assert blueprint.family == "linear"
        assert blueprint.question_types == ["gradient"]
