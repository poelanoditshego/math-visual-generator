import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

from ai.example_question_analyzer import ExampleQuestionAnalysis
from ai.image_question_analyzer import ImageQuestionAnalysis
from ai.question_writer import _validate_response, write_linear_question
from ai.transcript_analyzer import TranscriptAnalysis
from batch_loader import load_batch
from models.question_models import (
    SUPPORTED_LINEAR_QUESTION_TYPES,
    LinearQuestionData,
    QuestionBlueprint,
)
from questions.engine import generate_question_batch
from questions.linear import (
    generate_linear_question_batch,
)
from questions.persistence import save_question_batch
from questions.specs import LINEAR_QUESTION_SPECS, get_question_spec


class DrawLinearGraphTests(unittest.TestCase):
    def setUp(self):
        self.paths: list[Path] = []
        self.batch_directories: set[Path] = set()

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)
        for directory in self.batch_directories:
            shutil.rmtree(directory, ignore_errors=True)

    def test_1_registered_in_supported_types_and_specs(self):
        self.assertIn("draw_linear_graph", SUPPORTED_LINEAR_QUESTION_TYPES)
        self.assertIn("draw_linear_graph", LINEAR_QUESTION_SPECS)
        spec = get_question_spec("draw_linear_graph", family="linear")
        self.assertEqual(spec.question_type, "draw_linear_graph")
        self.assertEqual(spec.family, "linear")
        self.assertEqual(spec.answer_type, "graph")
        self.assertEqual(spec.graph_role, "memo")

    def test_2_generation_works_through_batch(self):
        blueprint = QuestionBlueprint(
            number_of_questions=3,
            question_types=["draw_linear_graph"],
            difficulty="Medium",
        )
        batch = generate_linear_question_batch(blueprint, seed=42)
        self.assertEqual(len(batch.questions), 3)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        for question in batch.questions:
            self.assertEqual(question.question_type, "draw_linear_graph")
            self.assertEqual(question.graph_role, "memo")

    def test_3_to_7_mathematical_data_and_equation(self):
        blueprint = QuestionBlueprint(
            number_of_questions=5,
            question_types=["draw_linear_graph"],
            difficulty="Easy",
        )
        batch = generate_linear_question_batch(blueprint, seed=101)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        for question in batch.questions:
            data = question.mathematical_data
            self.assertIsInstance(data, LinearQuestionData)
            # 3. equation is generated
            self.assertTrue(data.equation)
            # 4. gradient is non-zero integer
            self.assertIsInstance(data.gradient, int)
            self.assertNotEqual(data.gradient, 0)
            self.assertLessEqual(abs(data.gradient), 2)
            # 5. x-intercept is integer
            self.assertIsInstance(data.x_intercept, int)
            # 6. y-intercept is integer
            self.assertIsInstance(data.y_intercept, int)
            # 7. mathematical relationship holds: y_intercept = -gradient * x_intercept
            self.assertEqual(data.y_intercept, -data.gradient * data.x_intercept)
            self.assertIn(f"Draw the graph of f(x) = {data.equation.replace('*x', 'x')}.", question.question_text)

    def test_8_to_10_memo_contains_gradient_x_intercept_y_intercept(self):
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["draw_linear_graph"],
            difficulty="Easy",
        )
        batch = generate_linear_question_batch(blueprint, seed=202)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        q = batch.questions[0]
        data = q.mathematical_data
        # 8. memo contains gradient
        self.assertIn(f"m = {data.gradient}", q.memo)
        # 9. memo contains x-intercept
        self.assertIn(f"({data.x_intercept}; 0)", q.memo)
        # 10. memo contains y-intercept
        self.assertIn(f"(0; {data.y_intercept})", q.memo)

    def test_11_to_13_solution_graph_artifact_reuses_renderer(self):
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["draw_linear_graph"],
        )
        batch = generate_linear_question_batch(blueprint, seed=303)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        q = batch.questions[0]
        # 11. artifact exists
        graph_path = Path(q.graph_artifact.image_path)
        self.assertTrue(graph_path.is_file())
        self.assertGreater(graph_path.stat().st_size, 0)
        # 12. existing Linear renderer used
        self.assertEqual(q.graph_request.graph_type, "Linear")
        # 13. display settings for solution graph
        display = q.graph_request.display
        self.assertTrue(display.show_x_intercepts)
        self.assertTrue(display.show_y_intercepts)
        self.assertTrue(display.show_grid)
        self.assertTrue(display.show_tick_labels)
        self.assertTrue(display.show_tick_marks)
        self.assertFalse(display.show_gradient)

    def test_14_and_15_graph_role_presentation(self):
        spec = get_question_spec("draw_linear_graph", family="linear")
        # 14. learner-facing graph role is memo-only (not question)
        self.assertEqual(spec.graph_role, "memo")
        blueprint = QuestionBlueprint(
            number_of_questions=1,
            question_types=["draw_linear_graph"],
        )
        batch = generate_linear_question_batch(blueprint, seed=404)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        # 15. memo presentation keeps graph_role = "memo"
        self.assertEqual(batch.questions[0].graph_role, "memo")

    def test_16_ai_wording_preserves_equation(self):
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "question_text": "Draw the graph of f(x) = 2x - 4.",
            "memo": "y-intercept is (0; -4), x-intercept is (2; 0). Graph of f(x) = 2x - 4",
        })
        with patch("ai.question_writer.create_structured_response", return_value=mock_response):
            result = write_linear_question(
                grade=9,
                difficulty="Medium",
                question_type="draw_linear_graph",
                equation="2*x - 4",
                expected_answer="Graph of f(x) = 2x - 4",
                gradient=2,
                x_intercept=2,
                y_intercept=-4,
                visible_information=["equation"],
                hidden_information=["x-intercept", "y-intercept", "gradient"],
            )
            self.assertEqual(result.question_text, "Draw the graph of f(x) = 2x - 4.")
            self.assertIn("Graph of f(x) = 2x - 4", result.memo)

    def test_17_ai_wording_rejects_leakage(self):
        # AI reveals gradient in question text
        leaky_gradient = {
            "question_text": "Draw the graph of f(x) = 2x - 4 with gradient = 2.",
            "memo": "Graph of f(x) = 2x - 4",
        }
        with self.assertRaises(ValueError):
            _validate_response(
                leaky_gradient,
                question_type="draw_linear_graph",
                equation="2*x - 4",
                expected_answer="Graph of f(x) = 2x - 4",
                visible_information=["equation"],
                hidden_information=["x-intercept", "y-intercept", "gradient"],
            )

        # AI reveals x-intercept in question text
        leaky_intercept = {
            "question_text": "Draw the graph of f(x) = 2x - 4 with x-intercept is (2; 0).",
            "memo": "Graph of f(x) = 2x - 4",
        }
        with self.assertRaises(ValueError):
            _validate_response(
                leaky_intercept,
                question_type="draw_linear_graph",
                equation="2*x - 4",
                expected_answer="Graph of f(x) = 2x - 4",
                visible_information=["equation"],
                hidden_information=["x-intercept", "y-intercept", "gradient"],
            )

    def test_18_seed_reproducibility(self):
        blueprint = QuestionBlueprint(
            number_of_questions=3,
            question_types=["draw_linear_graph"],
        )
        first = generate_linear_question_batch(blueprint, seed=777)
        second = generate_linear_question_batch(blueprint, seed=777)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in first.questions + second.questions
        )
        self.assertEqual(
            [q.mathematical_data for q in first.questions],
            [q.mathematical_data for q in second.questions],
        )

    def test_19_uniqueness(self):
        blueprint = QuestionBlueprint(
            number_of_questions=10,
            question_types=["draw_linear_graph"],
            difficulty="Medium",
        )
        batch = generate_linear_question_batch(blueprint, seed=888)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        equations = [q.mathematical_data.equation for q in batch.questions]
        self.assertEqual(len(equations), len(set(equations)))

    def test_20_analyzer_schemas_include_type(self):
        from ai.example_question_analyzer import _ANALYSIS_SCHEMA as example_schema
        from ai.image_question_analyzer import _ANALYSIS_SCHEMA as image_schema
        from ai.transcript_analyzer import _TRANSCRIPT_SCHEMA as transcript_schema
        self.assertIn("draw_linear_graph", example_schema["properties"]["question_type"]["enum"])
        self.assertIn("draw_linear_graph", image_schema["properties"]["question_types"]["items"]["enum"])
        self.assertIn("draw_linear_graph", transcript_schema["properties"]["question_types"]["items"]["enum"])



    def test_21_batch_json_persistence_and_reloading(self):
        blueprint = QuestionBlueprint(
            number_of_questions=2,
            question_types=["draw_linear_graph"],
        )
        batch = generate_linear_question_batch(blueprint, seed=999)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        json_path = Path("generated_questions/test_draw_graph_batch.json")
        self.paths.append(json_path)
        save_question_batch(batch, json_path)

        loaded_data, error = load_batch(json_path.name)
        self.assertIsNone(error)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data["questions"]), 2)
        q_dict = loaded_data["questions"][0]
        self.assertEqual(q_dict["question_type"], "draw_linear_graph")
        self.assertEqual(q_dict["graph_role"], "memo")

    def test_22_existing_question_types_unaffected(self):
        blueprint = QuestionBlueprint(
            number_of_questions=5,
            question_types=["gradient", "x_intercept", "y_intercept", "parallel_lines"],
        )
        batch = generate_linear_question_batch(blueprint, seed=1234)
        self.batch_directories.update(
            Path(q.graph_artifact.image_path).parent for q in batch.questions
        )
        for q in batch.questions:
            self.assertEqual(q.graph_role, "question")


if __name__ == "__main__":
    unittest.main()
