import json
import os
from pathlib import Path
import shutil
import unittest

import matplotlib

matplotlib.use("Agg")

from models.question_models import (
    QuestionBlueprint,
    SUPPORTED_LINEAR_QUESTION_TYPES,
)
from questions.linear import (
    build_linear_display_settings,
    generate_linear_question_batch,
)
from questions.persistence import save_question_batch


class LinearQuestionTests(unittest.TestCase):
    def setUp(self):
        self.paths: list[Path] = []
        self.batch_directories: set[Path] = set()

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)
        for directory in self.batch_directories:
            shutil.rmtree(directory, ignore_errors=True)

    def test_blueprint_defaults_and_validation(self):
        blueprint = QuestionBlueprint()
        self.assertEqual(blueprint.number_of_questions, 10)
        blueprint.validate()
        for field_name, value in (
            ("number_of_questions", 0),
            ("marks_per_question", 0),
            ("grade", 0),
        ):
            invalid = QuestionBlueprint(**{field_name: value})
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                invalid.validate()
        with self.assertRaisesRegex(ValueError, "difficulty"):
            QuestionBlueprint(difficulty="Extreme").validate()
        with self.assertRaisesRegex(ValueError, "question type"):
            QuestionBlueprint(question_types=["quadratic"]).validate()

    def test_batch_has_requested_count_unique_fingerprints_and_integer_answers(self):
        blueprint = QuestionBlueprint(number_of_questions=10)
        batch = generate_linear_question_batch(blueprint, seed=123)
        self.assertEqual(len(batch.questions), 10)
        fingerprints = [
            (question.question_type, question.mathematical_data.gradient, question.mathematical_data.y_intercept)
            for question in batch.questions
        ]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        for question in batch.questions:
            data = question.mathematical_data
            self.assertEqual(data.x_intercept, -data.y_intercept / data.gradient)
            self.assertIn(question.question_type, SUPPORTED_LINEAR_QUESTION_TYPES)

    def test_seed_reproduces_mathematical_data(self):
        blueprint = QuestionBlueprint(number_of_questions=5)
        first = generate_linear_question_batch(blueprint, seed=123)
        second = generate_linear_question_batch(blueprint, seed=123)
        first_data = [question.mathematical_data for question in first.questions]
        second_data = [question.mathematical_data for question in second.questions]
        self.assertEqual(first_data, second_data)
        self.paths.extend(Path(question.graph_artifact.image_path) for question in first.questions)
        self.paths.extend(Path(question.graph_artifact.image_path) for question in second.questions)

    def test_separate_batches_isolate_same_sequential_graph_names(self):
        blueprint = QuestionBlueprint(number_of_questions=2)
        first = generate_linear_question_batch(blueprint, seed=123)
        second = generate_linear_question_batch(blueprint, seed=123)
        self.batch_directories.update(
            Path(question.graph_artifact.image_path).parent
            for question in first.questions + second.questions
        )

        first_paths = [Path(question.graph_artifact.image_path) for question in first.questions]
        second_paths = [Path(question.graph_artifact.image_path) for question in second.questions]
        self.assertNotEqual(first.batch_id, second.batch_id)
        self.assertEqual([path.name for path in first_paths], ["linear_0001.png", "linear_0002.png"])
        self.assertEqual([path.name for path in second_paths], ["linear_0001.png", "linear_0002.png"])
        self.assertNotEqual(first_paths[0].parent, second_paths[0].parent)
        self.assertTrue(all(path.is_file() for path in first_paths + second_paths))
        self.assertTrue(all(path.parent.name in {first.batch_id, second.batch_id} for path in first_paths + second_paths))

    def test_answer_hiding_settings(self):
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=6), seed=44
        )
        self.paths.extend(Path(question.graph_artifact.image_path) for question in batch.questions)
        for question in batch.questions:
            display = question.graph_request.display
            self.assertFalse(display.show_equation)
            self.assertFalse(display.show_title)
            self.assertFalse(display.show_legend)
            self.assertIn(
                f"f(x) = {question.mathematical_data.equation.replace('*x', 'x')}",
                question.question_text,
            )
            if question.question_type == "x_intercept":
                self.assertFalse(display.show_x_intercepts)
                self.assertTrue(display.show_y_intercepts)  # Show for reference
            elif question.question_type == "y_intercept":
                self.assertFalse(display.show_y_intercepts)
                self.assertTrue(display.show_x_intercepts)  # Show for reference
            else:
                self.assertFalse(display.show_gradient)
                self.assertFalse(display.show_gradient_triangle)
                self.assertTrue(display.show_x_intercepts)  # Show for reference
                self.assertTrue(display.show_y_intercepts)  # Show for reference

    def test_question_graph_does_not_rely_on_legend_for_equation_or_gradient(self):
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=3), seed=44
        )
        self.batch_directories.update(
            Path(question.graph_artifact.image_path).parent
            for question in batch.questions
        )
        for question in batch.questions:
            display = question.graph_request.display
            self.assertFalse(display.show_legend)
            self.assertFalse(display.show_title)
            self.assertFalse(display.show_equation)
            if question.question_type == "gradient":
                self.assertFalse(display.show_gradient)
                self.assertFalse(display.show_gradient_triangle)

    def test_display_planner_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            build_linear_display_settings("quadratic")

    def test_real_graph_artifact_is_created(self):
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=1, question_types=["x_intercept"]),
            seed=7,
        )
        question = batch.questions[0]
        self.paths.append(Path(question.graph_artifact.image_path))
        self.assertTrue(os.path.isfile(question.graph_artifact.image_path))
        self.assertGreater(os.path.getsize(question.graph_artifact.image_path), 0)

    def test_json_persistence_preserves_question_and_graph_request(self):
        batch = generate_linear_question_batch(
            QuestionBlueprint(number_of_questions=1), seed=5
        )
        graph_path = Path(batch.questions[0].graph_artifact.image_path)
        self.paths.append(graph_path)
        json_path = Path("generated_questions/test_linear_batch.json")
        self.paths.append(json_path)
        save_question_batch(batch, json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        question = payload["questions"][0]
        self.assertEqual(payload["blueprint"]["number_of_questions"], 1)
        self.assertEqual(question["question_text"], batch.questions[0].question_text)
        self.assertEqual(question["expected_answer"], batch.questions[0].expected_answer)
        self.assertEqual(question["mathematical_data"]["equation"], batch.questions[0].mathematical_data.equation)
        self.assertEqual(question["graph_request"]["graph_type"], "Linear")
        self.assertEqual(question["graph_artifact"]["image_path"], batch.questions[0].graph_artifact.image_path)


if __name__ == "__main__":
    unittest.main()
