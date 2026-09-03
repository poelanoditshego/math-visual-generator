from dataclasses import dataclass, field
from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.graph_artifact import GraphArtifact
from models.graph_request import GraphRequest

SUPPORTED_DIFFICULTIES = ("Easy", "Medium", "Hard")
SUPPORTED_LINEAR_QUESTION_TYPES = (
    "x_intercept",
    "y_intercept",
    "gradient",
    "determine_equation",
    "equation_from_two_points",
    "equation_from_gradient_and_point",
    "find_f_of_x",
    "find_x_given_y",
    "read_coordinate",
    "increasing_or_decreasing",
    "intersection_of_two_lines",
    "parallel_lines",
    "perpendicular_lines",
    "draw_linear_graph",
)



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
    family: str = "linear"

    def validate(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("Question family must be a non-empty string.")
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
        if self.family == "linear":
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
    input_x: int | float | None = None  # for find_f_of_x
    target_y: int | float | None = None  # for find_x_given_y
    selected_point: tuple[int | float, int | float] | None = None  # for read_coordinate
    point_a: tuple[int | float, int | float] | None = None
    point_b: tuple[int | float, int | float] | None = None
    second_equation: str | None = None
    second_gradient: int | float | None = None
    second_y_intercept: int | float | None = None
    intersection_point: tuple[int | float, int | float] | None = None

    @property
    def canonical_line_pair(self) -> tuple[tuple[int | float, int | float], ...]:
        """Return a line-order-independent identity for uniqueness checks."""
        if self.second_gradient is None or self.second_y_intercept is None:
            return ((self.gradient, self.y_intercept),)
        return tuple(
            sorted(
                (
                    (self.gradient, self.y_intercept),
                    (self.second_gradient, self.second_y_intercept),
                )
            )
        )

    @property
    def canonical_point_pair(self) -> tuple[tuple[int | float, int | float], ...]:
        """Return a point-order-independent identity for uniqueness checks."""
        if self.point_a is None or self.point_b is None:
            return tuple(point for point in (self.point_a, self.point_b) if point is not None)
        return tuple(sorted((self.point_a, self.point_b)))


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
    mathematical_data: object
    graph_request: GraphRequest
    graph_artifact: GraphArtifact
    graph_role: str = "question"


@dataclass
class QuestionBatch:
    blueprint: QuestionBlueprint
    questions: list[GeneratedQuestion]
    batch_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
