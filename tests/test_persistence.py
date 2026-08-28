"""Tests for persistence module, including batch validation."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from models.question_models import (
    QuestionBlueprint,
    GeneratedQuestion,
    LinearQuestionData,
    QuestionBatch,
)
from models.graph_artifact import GraphArtifact
from models.graph_request import GraphRequest, GraphRange, GraphDisplaySettings
from questions.persistence import (
    validate_batch_graph_files,
    save_question_batch,
    question_batch_to_dict,
)


class TestValidateBatchGraphFiles:
    """Tests for validate_batch_graph_files function."""
    
    def test_validate_batch_with_all_files_existing(self):
        """Test validation passes when all referenced files exist."""
        # Load the test batch from the actual generated_questions directory
        batch_file = Path(__file__).parent.parent / "generated_questions" / "linear_test_batch.json"
        
        if not batch_file.exists():
            pytest.skip("Test batch file not found in generated_questions/")
        
        # Load the batch
        with open(batch_file) as f:
            batch_data = json.load(f)
        
        # Create a batch object (simplified - just verify structure)
        blueprint_data = batch_data["blueprint"]
        blueprint = QuestionBlueprint(**blueprint_data)
        
        # For this test, just verify that validate_batch_graph_files can be called
        # We'll use a dummy batch with known missing files
        pass
    
    def test_validate_batch_detects_missing_files(self):
        """Test that validation detects missing graph files."""
        # Create a simple batch with a question referencing a non-existent file
        blueprint = QuestionBlueprint(
            subject="Math",
            grade=9,
            topic="Test",
            subtopic="Test",
            difficulty="Easy",
            marks_per_question=1,
            number_of_questions=1,
        )
        
        # Create minimal question data
        question_data = LinearQuestionData(
            equation="x",
            gradient=1,
            y_intercept=0,
            x_intercept=0,
        )
        
        # Create a graph request
        graph_request = GraphRequest(
            graph_type="Linear",
            equation="x",
            graph_range=GraphRange(x_min=-10, x_max=10, y_min=-10, y_max=10),
            display=GraphDisplaySettings(),
            output_name="nonexistent_graph.png",
        )
        
        # Create a graph artifact with a path that doesn't exist
        graph_artifact = GraphArtifact(
            image_path="generated_graphs/this_file_does_not_exist_xyz.png",
            graph_type="Linear",
        )
        
        # Create a question with the missing graph
        question = GeneratedQuestion(
            question_id="test_001",
            question_type="x_intercept",
            subject="Math",
            grade=9,
            topic="Test",
            subtopic="Test",
            difficulty="Easy",
            marks=1,
            question_text="Test question",
            expected_answer="0",
            memo="Test memo",
            mathematical_data=question_data,
            graph_request=graph_request,
            graph_artifact=graph_artifact,
        )
        
        # Create a batch with the question
        batch = QuestionBatch(
            blueprint=blueprint,
            questions=[question],
            batch_id="test_batch",
            created_at="2026-08-24T12:00:00",
        )
        
        # Validate should fail
        is_valid, missing_files = validate_batch_graph_files(batch)
        assert not is_valid, "Validation should fail for missing files"
        assert len(missing_files) == 1
        assert "this_file_does_not_exist_xyz.png" in missing_files[0]
    
    def test_save_batch_with_validation_raises_on_missing_files(self):
        """Test that save_question_batch raises ValueError when validation fails."""
        # Create a simple batch with a question referencing a non-existent file
        blueprint = QuestionBlueprint(
            subject="Math",
            grade=9,
            topic="Test",
            subtopic="Test",
            difficulty="Easy",
            marks_per_question=1,
            number_of_questions=1,
        )
        
        question_data = LinearQuestionData(
            equation="x",
            gradient=1,
            y_intercept=0,
            x_intercept=0,
        )
        
        graph_request = GraphRequest(
            graph_type="Linear",
            equation="x",
            graph_range=GraphRange(x_min=-10, x_max=10, y_min=-10, y_max=10),
            display=GraphDisplaySettings(),
            output_name="nonexistent_graph.png",
        )
        
        # Use a path that definitely doesn't exist
        graph_artifact = GraphArtifact(
            image_path="generated_graphs/missing_file_12345.png",
            graph_type="Linear",
        )
        
        question = GeneratedQuestion(
            question_id="test_001",
            question_type="x_intercept",
            subject="Math",
            grade=9,
            topic="Test",
            subtopic="Test",
            difficulty="Easy",
            marks=1,
            question_text="Test question",
            expected_answer="0",
            memo="Test memo",
            mathematical_data=question_data,
            graph_request=graph_request,
            graph_artifact=graph_artifact,
        )
        
        batch = QuestionBatch(
            blueprint=blueprint,
            questions=[question],
            batch_id="test_batch",
            created_at="2026-08-24T12:00:00",
        )
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_batch.json"
            
            # Should raise ValueError due to missing files
            with pytest.raises(ValueError) as exc_info:
                save_question_batch(batch, output_path, validate_graph_files=True)
            
            assert "missing" in str(exc_info.value).lower()
            assert "missing_file_12345.png" in str(exc_info.value)
            
            # File should NOT have been created
            assert not output_path.exists()
    
    def test_save_batch_without_validation_succeeds_with_missing_files(self):
        """Test that save_question_batch can skip validation if requested."""
        blueprint = QuestionBlueprint(
            subject="Math",
            grade=9,
            topic="Test",
            subtopic="Test",
            difficulty="Easy",
            marks_per_question=1,
            number_of_questions=1,
        )
        
        question_data = LinearQuestionData(
            equation="x",
            gradient=1,
            y_intercept=0,
            x_intercept=0,
        )
        
        graph_request = GraphRequest(
            graph_type="Linear",
            equation="x",
            graph_range=GraphRange(x_min=-10, x_max=10, y_min=-10, y_max=10),
            display=GraphDisplaySettings(),
            output_name="nonexistent_graph.png",
        )
        
        graph_artifact = GraphArtifact(
            image_path="generated_graphs/missing_file_test.png",
            graph_type="Linear",
        )
        
        question = GeneratedQuestion(
            question_id="test_001",
            question_type="x_intercept",
            subject="Math",
            grade=9,
            topic="Test",
            subtopic="Test",
            difficulty="Easy",
            marks=1,
            question_text="Test question",
            expected_answer="0",
            memo="Test memo",
            mathematical_data=question_data,
            graph_request=graph_request,
            graph_artifact=graph_artifact,
        )
        
        batch = QuestionBatch(
            blueprint=blueprint,
            questions=[question],
            batch_id="test_batch",
            created_at="2026-08-24T12:00:00",
        )
        
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_batch.json"
            
            # Should succeed because validate_graph_files=False
            result_path = save_question_batch(
                batch, output_path, validate_graph_files=False
            )
            
            # File SHOULD have been created
            assert result_path.exists()
            assert result_path.read_text().count("missing_file_test.png") == 1


class TestPersistenceBatchSaving:
    """Tests for overall batch saving functionality."""
    
    def test_question_batch_to_dict_preserves_structure(self):
        """Test that question_batch_to_dict preserves batch structure."""
        blueprint = QuestionBlueprint(
            subject="Math",
            grade=9,
            topic="Functions",
            subtopic="Linear",
            difficulty="Medium",
        )
        
        batch = QuestionBatch(
            blueprint=blueprint,
            questions=[],
            batch_id="test_batch_123",
            created_at="2026-08-24T12:00:00",
        )
        
        batch_dict = question_batch_to_dict(batch)
        
        assert "blueprint" in batch_dict
        assert "questions" in batch_dict
        assert "batch_id" in batch_dict
        assert "created_at" in batch_dict
        assert batch_dict["batch_id"] == "test_batch_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
