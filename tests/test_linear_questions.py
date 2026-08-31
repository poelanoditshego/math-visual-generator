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
            QuestionBlueprint(number_of_questions=6),
            seed=44
        )

        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )

        for question in batch.questions:
            display = question.graph_request.display

            self.assertFalse(display.show_equation)
            self.assertFalse(display.show_title)
            self.assertFalse(display.show_legend)

            equation_text = (
                f"f(x) = "
                f"{question.mathematical_data.equation.replace('*x', 'x')}"
            )

            if question.question_type in {
                "determine_equation", "read_coordinate", "increasing_or_decreasing"
            }:
                self.assertNotIn(
                    equation_text,
                    question.question_text,
                )
            else:
                self.assertIn(
                    equation_text,
                    question.question_text,
                )

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

    def test_determine_equation_is_supported(self):
        blueprint = QuestionBlueprint(
            question_types=["determine_equation"]
        )

        blueprint.validate()


    def test_determine_equation_generates_correct_answer(self):
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["determine_equation"],
        )

        batch = generate_linear_question_batch(
            blueprint,
            seed=123,
            use_ai=False,
        )

        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )

        question = batch.questions[0]

        expected_equation = question.mathematical_data.equation.replace("*x", "x")

        self.assertEqual(
            question.question_type,
            "determine_equation",
        )

        self.assertEqual(
            question.expected_answer,
            f"f(x) = {expected_equation}",
        )


    def test_determine_equation_does_not_use_origin_as_both_intercepts(self):
        blueprint = QuestionBlueprint(
            number_of_questions=5,
            question_types=["determine_equation"],
        )

        batch = generate_linear_question_batch(
            blueprint,
            seed=123,
            use_ai=False,
        )

        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )

        for question in batch.questions:
            self.assertNotEqual(
                question.mathematical_data.y_intercept,
                0,
            )

            self.assertNotEqual(
                question.mathematical_data.x_intercept,
                0,
            )


    def test_find_f_of_x_is_supported(self):
        blueprint = QuestionBlueprint(
            question_types=["find_f_of_x"]
        )
        blueprint.validate()

    def test_find_f_of_x_generates_correct_answer(self):
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["find_f_of_x"],
        )
        batch = generate_linear_question_batch(
            blueprint,
            seed=123,
            use_ai=False,
        )
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )
        question = batch.questions[0]
        data = question.mathematical_data

        self.assertEqual(question.question_type, "find_f_of_x")
        self.assertIsNotNone(data.input_x)

        # Verify answer: f(x) = mx + c
        expected_answer = data.gradient * data.input_x + data.y_intercept
        self.assertEqual(float(question.expected_answer), expected_answer)

    def test_find_f_of_x_stores_input_x(self):
        blueprint = QuestionBlueprint(
            number_of_questions=5,
            question_types=["find_f_of_x"],
        )
        batch = generate_linear_question_batch(blueprint, seed=456)
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )
        for question in batch.questions:
            self.assertIsNotNone(question.mathematical_data.input_x)

    def test_find_x_given_y_is_supported(self):
        blueprint = QuestionBlueprint(
            question_types=["find_x_given_y"]
        )
        blueprint.validate()

    def test_find_x_given_y_generates_correct_answer(self):
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["find_x_given_y"],
        )
        batch = generate_linear_question_batch(
            blueprint,
            seed=789,
            use_ai=False,
        )
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )
        question = batch.questions[0]
        data = question.mathematical_data

        self.assertEqual(question.question_type, "find_x_given_y")
        self.assertIsNotNone(data.target_y)
        self.assertIsNotNone(data.input_x)

        # Verify: target_y = mx + c, so answer is input_x
        expected_answer = data.input_x
        self.assertEqual(float(question.expected_answer), expected_answer)

    def test_find_x_given_y_stores_target_y(self):
        blueprint = QuestionBlueprint(
            number_of_questions=5,
            question_types=["find_x_given_y"],
        )
        batch = generate_linear_question_batch(blueprint, seed=999)
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )
        for question in batch.questions:
            self.assertIsNotNone(question.mathematical_data.target_y)

    def test_read_coordinate_is_supported(self):
        blueprint = QuestionBlueprint(
            question_types=["read_coordinate"]
        )
        blueprint.validate()

    def test_read_coordinate_stores_selected_point(self):
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["read_coordinate"],
        )
        batch = generate_linear_question_batch(blueprint, seed=111)
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )
        question = batch.questions[0]
        data = question.mathematical_data

        self.assertEqual(question.question_type, "read_coordinate")
        self.assertIsNotNone(data.selected_point)
        self.assertEqual(len(data.selected_point), 2)

    def test_read_coordinate_point_lies_on_line(self):
        blueprint = QuestionBlueprint(
            number_of_questions=5,
            question_types=["read_coordinate"],
        )
        batch = generate_linear_question_batch(blueprint, seed=222)
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )
        for question in batch.questions:
            data = question.mathematical_data
            x_val, y_val = data.selected_point

            # Verify point lies on the line: y = mx + c
            expected_y = data.gradient * x_val + data.y_intercept
            self.assertAlmostEqual(y_val, expected_y)
            graph_range = question.graph_request.graph_range
            self.assertGreater(x_val, graph_range.x_min)
            self.assertLess(x_val, graph_range.x_max)
            self.assertGreater(y_val, graph_range.y_min)
            self.assertLess(y_val, graph_range.y_max)
            self.assertEqual(question.expected_answer, f"({x_val}; {y_val})")
            self.assertEqual(question.graph_request.display.additional_point_labels, ["A"])
            self.assertNotIn(question.expected_answer, question.question_text)

    def test_increasing_or_decreasing_is_supported(self):
        blueprint = QuestionBlueprint(
            question_types=["increasing_or_decreasing"]
        )
        blueprint.validate()

    def test_increasing_or_decreasing_positive_gradient(self):
        # Generate many questions until we find one with positive gradient
        blueprint = QuestionBlueprint(
            number_of_questions=10,
            question_types=["increasing_or_decreasing"],
            difficulty="Easy",
        )
        batch = generate_linear_question_batch(blueprint, seed=333)
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )

        # Find a question with positive gradient
        for question in batch.questions:
            if question.mathematical_data.gradient > 0:
                self.assertEqual(question.expected_answer, "Increasing")
                self.assertNotIn("is increasing.", question.question_text.lower())
                break

    def test_increasing_or_decreasing_negative_gradient(self):
        # Generate many questions until we find one with negative gradient
        blueprint = QuestionBlueprint(
            number_of_questions=10,
            question_types=["increasing_or_decreasing"],
            difficulty="Easy",
        )
        batch = generate_linear_question_batch(blueprint, seed=444)
        self.paths.extend(
            Path(question.graph_artifact.image_path)
            for question in batch.questions
        )

        # Find a question with negative gradient
        for question in batch.questions:
            if question.mathematical_data.gradient < 0:
                self.assertEqual(question.expected_answer, "Decreasing")
                self.assertNotIn("is decreasing.", question.question_text.lower())
                break

    def test_new_types_all_supported_in_default_blueprint(self):
        blueprint = QuestionBlueprint()
        blueprint.validate()

        for new_type in ["find_f_of_x", "find_x_given_y", "read_coordinate", "increasing_or_decreasing"]:
            self.assertIn(new_type, blueprint.question_types)

    def test_seed_reproduces_new_type_data(self):
        blueprint = QuestionBlueprint(
            number_of_questions=3,
            question_types=["find_f_of_x", "find_x_given_y", "read_coordinate"],
        )
        first = generate_linear_question_batch(blueprint, seed=555)
        second = generate_linear_question_batch(blueprint, seed=555)

        self.paths.extend(Path(question.graph_artifact.image_path) for question in first.questions)
        self.paths.extend(Path(question.graph_artifact.image_path) for question in second.questions)

        first_data = [question.mathematical_data for question in first.questions]
        second_data = [question.mathematical_data for question in second.questions]
        self.assertEqual(first_data, second_data)

    def test_graph_exists_for_new_types(self):
        for question_type in ["find_f_of_x", "find_x_given_y", "read_coordinate", "increasing_or_decreasing"]:
            batch = generate_linear_question_batch(
                QuestionBlueprint(number_of_questions=1, question_types=[question_type]),
                seed=600,
            )
            self.paths.extend(
                Path(question.graph_artifact.image_path)
                for question in batch.questions
            )
            for question in batch.questions:
                graph_path = Path(question.graph_artifact.image_path)
                self.assertTrue(graph_path.is_file())
                self.assertGreater(graph_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
