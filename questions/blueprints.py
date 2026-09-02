"""Adapters from supported input analyses to the common question blueprint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from models.question_models import QuestionBlueprint


class SingleQuestionTypeAnalysis(Protocol):
    question_type: str


class MultipleQuestionTypesAnalysis(Protocol):
    question_types: Sequence[str]


def build_topic_question_blueprint(
    question_types: Sequence[str],
    *,
    grade: int,
    difficulty: str,
    number_of_questions: int,
) -> QuestionBlueprint:
    return QuestionBlueprint(
        grade=grade,
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        question_types=list(question_types),
    )


def build_example_question_blueprint(
    analysis: SingleQuestionTypeAnalysis,
    *,
    grade: int,
    difficulty: str,
    number_of_questions: int,
) -> QuestionBlueprint:
    return QuestionBlueprint(
        grade=grade,
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        question_types=[analysis.question_type],
    )


def build_transcript_question_blueprint(
    analysis: MultipleQuestionTypesAnalysis,
    *,
    grade: int,
    difficulty: str,
    number_of_questions: int,
) -> QuestionBlueprint:
    return QuestionBlueprint(
        grade=grade,
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        question_types=list(analysis.question_types),
    )


def build_image_question_blueprint(
    analysis: MultipleQuestionTypesAnalysis,
    *,
    grade: int,
    difficulty: str,
    number_of_questions: int,
) -> QuestionBlueprint:
    return QuestionBlueprint(
        grade=grade,
        difficulty=difficulty,
        number_of_questions=number_of_questions,
        question_types=list(analysis.question_types),
    )
