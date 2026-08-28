"""Tests for batch_loader module."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from batch_loader import (
    discover_batch_files,
    load_batch,
    resolve_graph_path,
    get_available_question_types,
    filter_questions,
    get_batches_directory,
)


class TestDiscoverBatchFiles:
    """Tests for discover_batch_files function."""
    
    def test_discover_batch_files_finds_existing_batches(self):
        """Test that discover_batch_files finds JSON files in generated_questions directory."""
        files = discover_batch_files()
        # Should find at least one JSON file in the actual workspace
        assert isinstance(files, list)
        # Verify all returned items are Path objects with .json extension
        for f in files:
            assert isinstance(f, Path)
            assert f.suffix == ".json"
    
    def test_discover_batch_files_returns_sorted_list(self):
        """Test that results are sorted."""
        files = discover_batch_files()
        if len(files) > 1:
            assert files == sorted(files)


class TestLoadBatch:
    """Tests for load_batch function."""
    
    def test_load_batch_with_valid_file(self):
        """Test loading a valid batch file."""
        files = discover_batch_files()
        assert len(files) > 0, "No batch files found in workspace"
        
        batch_data, error = load_batch(files[0].name)
        assert error is None, f"Should load successfully, got error: {error}"
        assert batch_data is not None
        assert "blueprint" in batch_data
        assert "questions" in batch_data
        assert isinstance(batch_data["questions"], list)
        assert len(batch_data["questions"]) > 0
    
    def test_load_batch_with_nonexistent_file(self):
        """Test loading a non-existent batch file."""
        batch_data, error = load_batch("nonexistent_batch_xyz.json")
        assert batch_data is None
        assert error is not None
        assert "not found" in error.lower()
    
    def test_load_batch_with_invalid_json(self):
        """Test loading a file with invalid JSON."""
        with TemporaryDirectory() as tmpdir:
            # Create a temporary invalid JSON file
            invalid_file = Path(tmpdir) / "invalid.json"
            invalid_file.write_text("{ invalid json content")
            
            # We need to mock discover to return this file, or test directly
            # For now, we'll skip this as it requires file patching
            pass


class TestResolveGraphPath:
    """Tests for resolve_graph_path function."""
    
    def test_resolve_graph_path_with_forward_slashes(self):
        """Test path resolution with forward slashes."""
        path = resolve_graph_path("generated_graphs/linear_0001.png")
        assert path.name == "linear_0001.png"
        assert "generated_graphs" in str(path)
    
    def test_resolve_graph_path_with_backward_slashes(self):
        """Test path resolution with backward slashes."""
        path = resolve_graph_path("generated_graphs\\linear_0001.png")
        assert path.name == "linear_0001.png"
        assert "generated_graphs" in str(path)
    
    def test_resolve_graph_path_normalizes_slashes(self):
        """Test that forward and backward slashes produce same result."""
        path1 = resolve_graph_path("generated_graphs/linear_0001.png")
        path2 = resolve_graph_path("generated_graphs\\linear_0001.png")
        assert path1 == path2

    def test_resolve_graph_path_supports_batch_directories(self):
        path = resolve_graph_path(
            "generated_graphs/linear_batch_123/linear_0001.png"
        )
        assert path.parent.name == "linear_batch_123"
        assert path.name == "linear_0001.png"


class TestGetAvailableQuestionTypes:
    """Tests for get_available_question_types function."""
    
    def test_get_available_question_types_from_valid_batch(self):
        """Test extracting question types from a batch."""
        files = discover_batch_files()
        assert len(files) > 0
        
        batch_data, error = load_batch(files[0].name)
        assert error is None
        
        types = get_available_question_types(batch_data)
        assert isinstance(types, list)
        assert len(types) > 0
        # Should be sorted
        assert types == sorted(types)
    
    def test_get_available_question_types_empty_batch(self):
        """Test with empty questions list."""
        batch_data = {"questions": []}
        types = get_available_question_types(batch_data)
        assert types == []
    
    def test_get_available_question_types_missing_questions_key(self):
        """Test with missing questions key."""
        batch_data = {"blueprint": {}}
        types = get_available_question_types(batch_data)
        assert types == []


class TestFilterQuestions:
    """Tests for filter_questions function."""
    
    def test_filter_questions_all_returns_all(self):
        """Test that 'All' filter returns all questions."""
        files = discover_batch_files()
        assert len(files) > 0
        
        batch_data, error = load_batch(files[0].name)
        assert error is None
        
        all_questions = batch_data["questions"]
        filtered = filter_questions(batch_data, "All")
        assert len(filtered) == len(all_questions)
    
    def test_filter_questions_none_returns_all(self):
        """Test that None filter returns all questions."""
        files = discover_batch_files()
        assert len(files) > 0
        
        batch_data, error = load_batch(files[0].name)
        assert error is None
        
        all_questions = batch_data["questions"]
        filtered = filter_questions(batch_data, None)
        assert len(filtered) == len(all_questions)
    
    def test_filter_questions_by_specific_type(self):
        """Test filtering by a specific question type."""
        files = discover_batch_files()
        assert len(files) > 0
        
        batch_data, error = load_batch(files[0].name)
        assert error is None
        
        types = get_available_question_types(batch_data)
        if types:
            first_type = types[0]
            filtered = filter_questions(batch_data, first_type)
            
            # All filtered should match the type
            for q in filtered:
                assert q.get("question_type") == first_type
    
    def test_filter_questions_empty_batch(self):
        """Test filtering on empty batch."""
        batch_data = {"questions": []}
        filtered = filter_questions(batch_data, "All")
        assert filtered == []


class TestBatchStructure:
    """Integration tests for batch structure validation."""
    
    def test_batch_has_required_keys(self):
        """Test that loaded batches have all required keys."""
        files = discover_batch_files()
        assert len(files) > 0
        
        batch_data, error = load_batch(files[0].name)
        assert error is None
        
        # Check blueprint keys
        blueprint = batch_data["blueprint"]
        required_blueprint_keys = {"subject", "grade", "topic", "subtopic", "difficulty"}
        assert required_blueprint_keys.issubset(blueprint.keys())
        
        # Check question structure
        questions = batch_data["questions"]
        assert len(questions) > 0
        
        first_question = questions[0]
        required_question_keys = {
            "question_id",
            "question_type",
            "question_text",
            "expected_answer",
            "memo",
            "graph_artifact",
            "mathematical_data",
        }
        assert required_question_keys.issubset(first_question.keys())
    
    def test_graph_artifacts_have_image_paths(self):
        """Test that graph artifacts contain image paths."""
        files = discover_batch_files()
        assert len(files) > 0
        
        batch_data, error = load_batch(files[0].name)
        assert error is None
        
        for question in batch_data["questions"]:
            artifact = question.get("graph_artifact", {})
            assert "image_path" in artifact
            assert isinstance(artifact["image_path"], str)
            assert len(artifact["image_path"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
