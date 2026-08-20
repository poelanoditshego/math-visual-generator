from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.graph_artifact import GraphArtifact
from models.graph_request import GraphRequest

SUPPORTED_DIFFICULTIES = ("Easy", "Medium", "Hard")
SUPPORTED_LINEAR_QUESTION_TYPES = ("x_intercept", "y_intercept", "gradient")


@dataclass
class QuestionBlueprint:
    subject: str = "Mathematics"
    grade: int = 9
    topic: str = "Functions"
    subtopic: str = "Linear Functions"
    difficulty: str = "Medium"
    marks_per_question: int = 4
    number_of_questions: int = 10
    question_types: list[str] = field(
        default_factory=lambda: list(SUPPORTED_LINEAR_QUESTION_TYPES)
    )

    def validate(self) -> None:
        if not isinstance(self.grade, int) or isinstance(self.grade, bool) or self.grade <= 0:
            raise ValueError("Grade must be a positive integer.")
        if (
            not isinstance(self.marks_per_question, int)
            or isinstance(self.marks_per_question, bool)
            or self.marks_per_question <= 0
        ):
            raise ValueError("Marks per question must be a positive integer.")
        if (
            not isinstance(self.number_of_questions, int)
            or isinstance(self.number_of_questions, bool)
            or self.number_of_questions <= 0
        ):
            raise ValueError("Number of questions must be a positive integer.")
        if self.difficulty not in SUPPORTED_DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {self.difficulty}")
        if not self.question_types:
            raise ValueError("At least one question type is required.")
        unsupported = set(self.question_types) - set(SUPPORTED_LINEAR_QUESTION_TYPES)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported linear question type: {names}")


@dataclass(frozen=True)
class LinearQuestionData:
    equation: str
    gradient: int | float
    y_intercept: int | float
    x_intercept: int | float | None
    function_name: str = "f"


@dataclass
class GeneratedQuestion:
    question_id: str
    question_type: str
    subject: str
    grade: int
    topic: str
    subtopic: str
    difficulty: str
    marks: int
    question_text: str
    expected_answer: str
    memo: str
    mathematical_data: LinearQuestionData
    graph_request: GraphRequest
    graph_artifact: GraphArtifact


@dataclass
class QuestionBatch:
    blueprint: QuestionBlueprint
    questions: list[GeneratedQuestion]
    batch_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
