import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from models.question_models import QuestionBlueprint
from questions.linear import generate_linear_question_batch
from ai.question_writer import write_linear_question


def _ai_result(question_text="AI question", memo="AI memo (1, 0)"):
    return SimpleNamespace(
        output_text=json.dumps({"question_text": question_text, "memo": memo})
    )


def _writer_result(**kwargs):
    return SimpleNamespace(
        question_text=f"AI question for {kwargs['question_type']}",
        memo=f"AI method. Therefore, the answer is {kwargs['expected_answer']}",
    )


def test_use_ai_false_does_not_call_writer():
    with patch("questions.linear.write_linear_question") as writer:
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=1, question_types=["x_intercept"]),
            seed=7,
        )
    writer.assert_not_called()
    shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_use_ai_uses_text_and_memo_but_preserves_python_answer_and_display():
    with patch("questions.linear.write_linear_question", side_effect=_writer_result) as writer:
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=1, question_types=["x_intercept"]),
            seed=7,
            use_ai=True,
        )
    question = batch.questions[0]
    assert question.question_text == "AI question for x_intercept"
    assert question.memo.endswith(question.expected_answer)
    assert question.expected_answer == "(-3, 0)"
    assert question.graph_request.display.show_title is False
    assert question.graph_request.display.show_legend is False
    writer.assert_called_once()
    shutil.rmtree(Path(question.graph_artifact.image_path).parent)


def test_ai_exception_falls_back_for_one_question():
    with patch(
        "questions.linear.write_linear_question",
        side_effect=RuntimeError("service unavailable"),
    ):
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=1, question_types=["x_intercept"]),
            seed=7,
            use_ai=True,
        )
    question = batch.questions[0]
    assert question.question_text.startswith("The graph of f(x) =")
    assert question.expected_answer == "(-3, 0)"
    shutil.rmtree(Path(question.graph_artifact.image_path).parent)


def test_seeded_mathematics_is_unchanged_when_ai_wording_changes():
    blueprint = QuestionBlueprint(number_of_questions=3)
    deterministic = generate_linear_question_batch(blueprint, seed=123)
    with patch("questions.linear.write_linear_question", side_effect=_writer_result):
        ai_batch = generate_linear_question_batch(blueprint, seed=123, use_ai=True)
    deterministic_data = [q.mathematical_data for q in deterministic.questions]
    ai_data = [q.mathematical_data for q in ai_batch.questions]
    assert deterministic_data == ai_data
    for batch in (deterministic, ai_batch):
        shutil.rmtree(Path(batch.questions[0].graph_artifact.image_path).parent)


def test_ai_client_module_does_not_create_client_on_import():
    with patch("ai.client.OpenAI") as openai:
        import ai.client
        importlib.reload(ai.client)
    openai.assert_not_called()


def test_invalid_structured_response_is_retried_once():
    valid = _ai_result(
        "The graph of f(x) = 2x + 6 is shown below. Determine the x-intercept of the graph.",
        "Set y = 0: 0 = 2x + 6, so x = -3. Therefore, the x-intercept is (-3, 0)",
    )
    with patch(
        "ai.question_writer.create_structured_response",
        side_effect=[_ai_result("TODO", "TODO"), valid],
    ) as request:
        result = write_linear_question(
            grade=9,
            difficulty="Medium",
            question_type="x_intercept",
            equation="2*x + 6",
            expected_answer="(-3, 0)",
            gradient=2,
            x_intercept=-3,
            y_intercept=6,
            visible_information=["equation", "y-intercept"],
            hidden_information=["x-intercept"],
        )
    assert result.question_text == json.loads(valid.output_text)["question_text"]
    assert result.memo == json.loads(valid.output_text)["memo"]
    assert request.call_count == 2
