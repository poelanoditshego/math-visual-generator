"""Helper functions for loading and managing question batches."""

import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from models.question_models import QuestionBatch, GeneratedQuestion, QuestionBlueprint


def get_batches_directory() -> Path:
    """Get the path to the generated_questions directory."""
    return Path(__file__).parent / "generated_questions"


def discover_batch_files() -> list[Path]:
    """
    Discover all available question batch JSON files.
    
    Returns:
        List of Path objects for all .json files in generated_questions/
    """
    batches_dir = get_batches_directory()
    if not batches_dir.exists():
        return []
    
    return sorted(batches_dir.glob("*.json"))


def load_batch_from_file(file_path: Path) -> tuple[Optional[QuestionBatch], Optional[str]]:
    """
    Load a question batch from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (QuestionBatch or None, error_message or None)
        Returns (batch, None) on success or (None, error_message) on failure
    """
    try:
        if not file_path.exists():
            return None, f"File not found: {file_path}"
        
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in {file_path.name}: {str(e)}"
    except Exception as e:
        return None, f"Error reading file {file_path.name}: {str(e)}"
    
    try:
        # Validate basic structure
        if "blueprint" not in data:
            return None, "Batch JSON missing 'blueprint' key"
        
        if "questions" not in data:
            return None, "Batch JSON missing 'questions' key"
        
        if not isinstance(data["questions"], list):
            return None, "'questions' must be a list"
        
        if len(data["questions"]) == 0:
            return None, "Batch contains no questions"
        
        # For now, we'll accept the batch as-is without full validation
        # since the structure has been pre-validated during generation
        batch = QuestionBatch(
            blueprint=QuestionBlueprint(**data["blueprint"]),
            questions=[],  # We'll keep raw question data for serialization
            batch_id=data.get("batch_id", file_path.stem),
            created_at=data.get("created_at", ""),
        )
        
        # Store raw data for display
        batch._raw_questions = data["questions"]
        
        return batch, None
    except Exception as e:
        return None, f"Error parsing batch: {str(e)}"


def load_batch(file_name: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Load a batch by filename (convenience wrapper).
    
    Args:
        file_name: Name of the JSON file (e.g., "linear_test_batch.json")
        
    Returns:
        Tuple of (batch_dict or None, error_message or None)
    """
    batches_dir = get_batches_directory()
    file_path = batches_dir / file_name
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict) or "questions" not in data:
            return None, "Invalid batch structure"
        
        if not isinstance(data["questions"], list) or len(data["questions"]) == 0:
            return None, "Batch has no questions"
        
        return data, None
    except FileNotFoundError:
        return None, f"Batch file not found: {file_name}"
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return None, f"Error loading batch: {str(e)}"


def resolve_graph_path(image_path: str) -> Path:
    """
    Resolve a graph image path relative to the project root.
    
    Handles both forward and backward slashes.
    
    Args:
        image_path: Path string from the JSON (e.g., "generated_graphs/batch-id/linear_0001.png")
        
    Returns:
        Absolute Path object
    """
    normalized = image_path.replace("\\", "/")
    
    project_root = Path(__file__).parent
    return project_root / Path(*normalized.split("/"))


def get_available_question_types(batch_data: dict) -> list[str]:
    """
    Extract unique question types from a batch.
    
    Args:
        batch_data: Loaded batch dictionary
        
    Returns:
        Sorted list of unique question types
    """
    types = set()
    for question in batch_data.get("questions", []):
        question_type = question.get("question_type")
        if question_type:
            types.add(question_type)
    
    return sorted(list(types))


def filter_questions(batch_data: dict, question_type: Optional[str] = None) -> list[dict]:
    """
    Filter questions in a batch by type.
    
    Args:
        batch_data: Loaded batch dictionary
        question_type: Type to filter by, or None for all questions
        
    Returns:
        List of filtered question dictionaries
    """
    questions = batch_data.get("questions", [])
    
    if question_type is None or question_type == "All":
        return questions
    
    return [q for q in questions if q.get("question_type") == question_type]
