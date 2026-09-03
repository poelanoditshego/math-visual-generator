"""Reusable orchestration for deterministic question-family generators."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from generators.api import generate_graph_from_request
from models.graph_request import GraphRequest
from models.question_models import GeneratedQuestion, QuestionBatch, QuestionBlueprint
from questions.specs import QuestionSpec, get_question_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionCandidate:
    """Family-produced content awaiting generic graph and record creation."""

    mathematical_data: Any
    fingerprint: tuple[Any, ...]
    graph_request: GraphRequest
    question_text: str
    expected_answer: str
    memo: str


class QuestionFamilyGenerator(Protocol):
    """Small plug-in boundary between orchestration and family mathematics."""

    family: str
    batch_prefix: str

    def create_candidate(
        self,
        *,
        rng: random.Random,
        difficulty: str,
        spec: QuestionSpec,
        output_name: str,
    ) -> QuestionCandidate | None: ...

    def rewrite_with_ai(
        self,
        *,
        blueprint: QuestionBlueprint,
        spec: QuestionSpec,
        candidate: QuestionCandidate,
    ) -> tuple[str, str]: ...


_FAMILY_GENERATORS: dict[str, QuestionFamilyGenerator] = {}


def register_family_generator(generator: QuestionFamilyGenerator) -> None:
    """Register or replace a family generator under its stable family name."""
    _FAMILY_GENERATORS[generator.family] = generator


def get_family_generator(family: str) -> QuestionFamilyGenerator:
    """Resolve a registered family generator."""
    if family not in _FAMILY_GENERATORS and family == "linear":
        # Importing the built-in family performs its explicit registration.
        from questions import linear  # noqa: F401

    try:
        return _FAMILY_GENERATORS[family]
    except KeyError as error:
        raise ValueError(f"Unsupported question family: {family}") from error


def generate_question_batch(
    blueprint: QuestionBlueprint,
    *,
    seed: int | None = None,
    use_ai: bool = False,
    family_generator: QuestionFamilyGenerator | None = None,
) -> QuestionBatch:
    """Generate a batch using generic orchestration and family-owned mathematics."""
    blueprint.validate()
    generator = family_generator or get_family_generator(blueprint.family)
    if generator.family != blueprint.family:
        raise ValueError(
            f"Blueprint family {blueprint.family!r} does not match generator "
            f"family {generator.family!r}."
        )

    specs = [
        get_question_spec(question_type, family=blueprint.family)
        for question_type in blueprint.question_types
    ]
    batch_id = f"{generator.batch_prefix}_{uuid4().hex}"
    batch_output_directory = Path("generated_graphs") / batch_id
    batch_output_directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    fingerprints: set[tuple[Any, ...]] = set()
    questions: list[GeneratedQuestion] = []
    max_attempts = blueprint.number_of_questions * 5
    attempts = 0

    while len(questions) < blueprint.number_of_questions and attempts < max_attempts:
        attempts += 1
        spec = rng.choice(specs)
        sequence = len(questions) + 1
        output_name = f"{generator.batch_prefix}_{sequence:04d}.png"
        candidate = generator.create_candidate(
            rng=rng,
            difficulty=blueprint.difficulty,
            spec=spec,
            output_name=output_name,
        )
        if candidate is None or candidate.fingerprint in fingerprints:
            continue

        question_id = f"{generator.batch_prefix}_{sequence:04d}"
        question_text = candidate.question_text
        memo = candidate.memo
        if use_ai:
            try:
                question_text, memo = generator.rewrite_with_ai(
                    blueprint=blueprint,
                    spec=spec,
                    candidate=candidate,
                )
                logger.info("AI wording generated for %s", question_id)
            except Exception as error:
                logger.warning(
                    "AI request failed for %s; using deterministic fallback",
                    question_id,
                )
                logger.error(
                    "AI error for %s: %s: %s",
                    question_id,
                    type(error).__name__,
                    error,
                )

        artifact = generate_graph_from_request(
            candidate.graph_request,
            output_directory=batch_output_directory,
        )
        questions.append(
            GeneratedQuestion(
                question_id=question_id,
                question_type=spec.question_type,
                subject=blueprint.subject,
                grade=blueprint.grade,
                topic=blueprint.topic,
                subtopic=blueprint.subtopic,
                difficulty=blueprint.difficulty,
                marks=blueprint.marks_per_question,
                question_text=question_text,
                expected_answer=candidate.expected_answer,
                memo=memo,
                mathematical_data=candidate.mathematical_data,
                graph_request=candidate.graph_request,
                graph_artifact=artifact,
                graph_role=spec.graph_role,
            )
        )

        fingerprints.add(candidate.fingerprint)

    if len(questions) != blueprint.number_of_questions:
        raise ValueError(
            f"Could not generate {blueprint.number_of_questions} unique questions; "
            f"generated {len(questions)} after {attempts} attempts."
        )

    return QuestionBatch(
        blueprint=blueprint,
        questions=questions,
        batch_id=batch_id,
    )
