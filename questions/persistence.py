from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from batch_loader import resolve_graph_path
from models.question_models import QuestionBatch


def question_batch_to_dict(batch: QuestionBatch) -> dict[str, object]:
    return asdict(batch)


def validate_batch_graph_files(batch: QuestionBatch) -> tuple[bool, list[str]]:
    """
    Validate that all graph files referenced in a batch exist on disk.
    
    Args:
        batch: The question batch to validate
        
    Returns:
        Tuple of (is_valid, missing_files_list)
        is_valid: True if all files exist, False otherwise
        missing_files_list: List of missing file paths (empty if all exist)
    """
    missing_files = []
    
    for question in batch.questions:
        graph_artifact = question.graph_artifact
        image_path_str = graph_artifact.image_path
        
        file_path = resolve_graph_path(image_path_str)
        
        if not file_path.is_file():
            missing_files.append(image_path_str)
    
    return len(missing_files) == 0, missing_files


def save_question_batch(
    batch: QuestionBatch,
    output_path: str | Path,
    *,
    validate_graph_files: bool = True,
) -> Path:
    """
    Save a question batch to JSON file.
    
    Args:
        batch: The question batch to save
        output_path: Path where the JSON file will be saved
        validate_graph_files: If True, validate that all referenced graph files exist
                            before saving. Raises ValueError if validation fails.
        
    Returns:
        Path object pointing to the saved file
        
    Raises:
        ValueError: If validate_graph_files is True and any graph files are missing
    """
    if validate_graph_files:
        is_valid, missing_files = validate_batch_graph_files(batch)
        if not is_valid:
            missing_str = "\n  - ".join(missing_files)
            raise ValueError(
                f"Cannot save batch: {len(missing_files)} graph file(s) are missing:\n  - {missing_str}"
            )
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(question_batch_to_dict(batch), indent=2),
        encoding="utf-8",
    )
    return path
