from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from models.question_models import QuestionBatch


def question_batch_to_dict(batch: QuestionBatch) -> dict[str, object]:
    return asdict(batch)


def save_question_batch(
    batch: QuestionBatch,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(question_batch_to_dict(batch), indent=2),
        encoding="utf-8",
    )
    return path
