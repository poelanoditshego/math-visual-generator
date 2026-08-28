# Math Visual Generator: Technical Handover

## 1. Scope and current state

This document describes the repository as inspected on 2026-08-28. The project is a Python/Matplotlib/SymPy application for generating mathematical graphs and deterministic Linear question batches, with JSON persistence, a Streamlit graph UI, a Streamlit read-only question reviewer, and an optional OpenAI wording/memo layer.

The current implemented AI phase writes only `question_text` and `memo`. Python remains authoritative for all mathematical values, graph settings, graph files, expected answers, and persistence.

## 2. Repository structure

Important source tree:

```text
math-visual-generator/
├── app.py
├── batch_loader.py
├── constants.py
├── main.py
├── README.md
├── requirements.txt
├── review_questions.py
├── run_linear_questions.py
├── test_openai.py
├── ai/
│   ├── __init__.py
│   ├── client.py
│   └── question_writer.py
├── generators/
│   ├── __init__.py
│   ├── api.py
│   ├── circle.py
│   ├── cosine.py
│   ├── cubic.py
│   ├── exponential.py
│   ├── graph_helpers.py
│   ├── hyperbola.py
│   ├── linear.py
│   ├── logarithmic.py
│   ├── mixed.py
│   ├── quadratic.py
│   ├── sine.py
│   ├── tangent.py
│   └── trig_helpers.py
├── models/
│   ├── __init__.py
│   ├── graph_artifact.py
│   ├── graph_request.py
│   ├── graph_settings.py
│   └── question_models.py
├── questions/
│   ├── __init__.py
│   ├── linear.py
│   └── persistence.py
└── tests/
    ├── test_ai_question_writer.py
    ├── test_batch_loader.py
    ├── test_cartesian_axes.py
    ├── test_display_controls.py
    ├── test_graph_api.py
    ├── test_graph_request.py
    ├── test_linear_display_policy.py
    ├── test_linear_questions.py
    ├── test_mixed.py
    ├── test_parsing.py
    └── test_persistence.py
```

Generated-data folders:

```text
generated_graphs/
├── standalone/manual PNG files such as linear_graph.png and custom_linear.png
└── linear_<uuid>/
    ├── linear_0001.png
    ├── linear_0002.png
    └── ...
generated_questions/
└── linear_test_batch.json
```

The current workspace also contains multiple historical/generated `linear_<uuid>` directories. They are data artifacts, not Python modules. Cache folders such as `__pycache__`, `.git`, and pytest caches are intentionally omitted above.

## 3. Entry points

### `app.py`

The manual Streamlit graph generator. It exposes controls for graph type, equations, ranges, labels, axes, display options, and output filename. It calls `generators.api.generate_graph()` directly. Its settings preserve manual-user behavior and are independent of the Linear question display policy.

Run on Windows with Python 3.11:

```powershell
py -3.11 -m streamlit run app.py
```

### `review_questions.py`

The read-only Streamlit question-batch reviewer. It discovers JSON files in `generated_questions`, loads raw JSON, displays batch metadata and questions, resolves nested graph paths, and provides filtering/navigation plus Answer, Memo, and Admin Details expanders.

Run:

```powershell
py -3.11 -m streamlit run review_questions.py
```

### `run_linear_questions.py`

The command-line Linear batch runner. It currently builds a Grade 9 Medium batch with `number_of_questions=3`, seed `123`, and passes `use_ai` from the environment. It saves to `generated_questions/linear_test_batch.json` with graph validation enabled and prints each question and graph path.

Run deterministic mode:

```powershell
py -3.11 run_linear_questions.py
```

Enable AI mode:

```powershell
$env:USE_AI="true"
py -3.11 run_linear_questions.py
```

`USE_AI` defaults to false. The runner can be changed back to 10 questions by editing its `QuestionBlueprint` construction; the generator itself supports arbitrary valid counts.

### `main.py`

Small interactive console entry point for one manual Linear or Quadratic graph. It calls graph-specific generators directly rather than the structured graph API. Run with:

```powershell
py -3.11 main.py
```

### `test_openai.py`

Opt-in live OpenAI smoke test. It makes no API call during import or ordinary pytest runs. Run it only when credentials and network are intentionally available:

```powershell
$env:RUN_OPENAI_SMOKE="true"
py -3.11 -m pytest test_openai.py
```

## 4. Graph architecture

### Supported graph types

`constants.SUPPORTED_GRAPH_TYPES` is the tuple:

```text
Linear, Quadratic, Exponential, Hyperbola, Cubic, Logarithmic,
Sine, Cosine, Tangent, Circle, Mixed
```

`generators.api._SINGLE_EQUATION_GENERATORS` maps every non-Mixed type to its graph function:

| Type | Module/function |
|---|---|
| Linear | `generators.linear.create_linear_graph(equation, settings)` |
| Quadratic | `generators.quadratic.create_quadratic_graph(equation, settings)` |
| Exponential | `generators.exponential.create_exponential_graph(equation, settings)` |
| Hyperbola | `generators.hyperbola.create_hyperbola_graph(equation, settings)` |
| Cubic | `generators.cubic.create_cubic_graph(equation, settings)` |
| Logarithmic | `generators.logarithmic.create_logarithmic_graph(equation, settings)` |
| Sine | `generators.sine.create_sine_graph(equation, settings)` |
| Cosine | `generators.cosine.create_cosine_graph(equation, settings)` |
| Tangent | `generators.tangent.create_tangent_graph(equation, settings)` |
| Circle | `generators.circle.create_circle_graph(equation, settings)` |
| Mixed | `generators.mixed.create_mixed_graph(equations, settings)` |

All are implemented sufficiently for the current tests. The graph generators write PNGs and generally return `None`; the API wraps the result in a `GraphArtifact` after checking the file exists.

### Models

`models.graph_settings.GraphSettings` is the internal, mutable rendering configuration. Its fields include:

- ranges: `x_min`, `x_max`, `y_min`, `y_max`;
- graph text: `title`, `x_label`, `y_label`;
- display: `show_grid`, `show_axes`, `show_equation`, `show_title`, `show_legend`, `show_border`, `show_tick_marks`, `show_tick_labels`, `show_axis_arrows`, `show_axis_labels`, `show_graph_arrows`;
- labels: `point_label_style`, `graph_label_style`, `graph_curve_label_style`, `axis_intercept_label_style`, `show_point_labels`, `show_origin_label`;
- Linear features: `show_x_intercepts`, `show_y_intercepts`, `show_gradient`, `show_gradient_triangle`;
- polynomial/asymptote/trig/circle feature toggles;
- output: `output_name`, `output_directory`, `image_dpi`;
- additional coordinate and annotation settings.

The default `output_directory` is `Path("generated_graphs")`, preserving standalone behavior. Batch generation supplies a nested directory.

`models.graph_request.GraphRange` is frozen and holds `x_min`, `x_max`, `y_min`, `y_max`. `validate()` requires finite numeric values and strictly increasing min/max pairs.

`GraphDisplaySettings` is the public request-level subset of display controls. It includes grid/axes/equation/title/legend controls, border/tick controls, point/intercept controls, gradient controls, asymptote/stationary/trigonometric/circle controls, and label styles. `validate()` checks supported label styles and trig angle mode.

`GraphRequest` contains:

```python
@dataclass
class GraphRequest:
    graph_type: str
    equation: str | None = None
    equations: list[str] | None = None
    graph_range: GraphRange = field(default_factory=GraphRange)
    display: GraphDisplaySettings = field(default_factory=GraphDisplaySettings)
    output_name: str = "graph.png"
```

It validates graph type, range, display settings, safe PNG filename, and the equation/equations contract. Mixed requests require exactly two non-empty equations and no single `equation` field.

`models.graph_artifact.GraphArtifact` is frozen:

```python
@dataclass(frozen=True)
class GraphArtifact:
    image_path: str
    graph_type: str
```

### API flow

`generators.api.graph_request_to_settings(request, *, output_directory=None)` copies public request controls into `GraphSettings`, including title/legend and output directory. The default directory is `generated_graphs`.

`generate_graph(graph_type, settings, equation=None, equations=None)`:

1. validates graph type and `settings.output_name`;
2. validates the equation channel for Mixed or non-Mixed graphs;
3. calls the mapped graph generator;
4. checks `Path(settings.output_directory) / settings.output_name` exists;
5. returns `GraphArtifact(image_path=str(path), graph_type=graph_type)`.

`generate_graph_from_request(request, *, output_directory=None)` validates the request, converts it to settings, and delegates to `generate_graph()`.

### Rendering controls

- Title: `show_title`; Linear renderer uses `ax.set_title(...)` only when true.
- Legend: `show_legend`, combined with equation/label settings by `graph_legend_is_enabled()` in `generators.graph_helpers`.
- Equation: `show_equation`; it supplies curve labels but does not itself force a legend unless the legend policy enables one.
- Intercepts: `show_x_intercepts`, `show_y_intercepts`, and shared point-label settings.
- Point labels: `show_point_labels`, `point_label_style`, and `axis_intercept_label_style`.
- Gradient: Linear `show_gradient` adds a legend-only gradient entry; `show_gradient_triangle` draws run/rise lines and labels.
- Asymptotes: graph-specific toggles such as `show_vertical_asymptote`, `show_horizontal_asymptote`, and `show_asymptote_labels`.
- Axes/ranges: `show_axes`, axis style/arrow/label/tick fields, and the four `GraphRange` values.

## 5. Graph type status and Mixed behavior

| Type | Status | Notes |
|---|---|---|
| Linear | Implemented | Polynomial degree 1; full generator and question flow. |
| Quadratic | Implemented | Polynomial degree 2; turning point and axis-of-symmetry features. |
| Exponential | Implemented | Supported exponential forms and horizontal asymptote. |
| Hyperbola | Implemented | Hyperbola parameter extraction and asymptote/centre features. |
| Cubic | Implemented | Stationary and inflection-point features. |
| Logarithmic | Implemented | Domain-aware logarithmic plotting. |
| Sine | Implemented | Trigonometric modes and key-point options. |
| Cosine | Implemented | Shared trig helpers. |
| Tangent | Implemented | Period/asymptote-aware trig plotting. |
| Circle | Implemented | Circle equation parsing, centre/radius/intercepts/cardinal points. |
| Mixed | Implemented with limits | Exactly two equations. Each is classified as Linear, Quadratic, supported Exponential, or Hyperbola. Other types/combinations are rejected. |

Mixed detection is in `generators.mixed.detect_graph_type()`. It checks supported exponential structure, polynomial degree 1/2, then hyperbola parameter extraction. Mixed rendering supports the detected combinations and shared intersection/label logic; it does not support cubic, logarithmic, sine, cosine, tangent, or circle expressions as Mixed members.

## 6. Parsing and validation

The shared parser is `generators.graph_helpers.parse_arithmetic_expression(...)`. It:

- accepts an expression such as `2*x - 4`;
- accepts `y = expression` when the left side is exactly `y` (case-insensitive);
- restricts characters using a safe expression regex;
- identifies identifiers and function calls;
- rejects unsupported variables/functions;
- uses SymPy `sympify` with an `x` symbol and explicitly allowed functions/constants;
- rejects free symbols other than `x`;
- converts SymPy/parser failures into `ValueError`.

`validate_polynomial_degree(expression, x, graph_name, degree)` requires a polynomial of the exact degree. Specialized modules provide parsers for exponential, logarithmic, trigonometric, hyperbola, and circle expressions.

Circle-specific parsing is `generators.circle.parse_circle_equation()`, producing `CircleParameters` with centre/radius/diameter properties. Hyperbola and trigonometric modules have corresponding parameter parsers.

`models.graph_request.validate_output_name()` accepts only a `.png` filename matching letters, numbers, spaces, hyphens, and underscores. It intentionally rejects path separators and traversal, so nested directories are supplied separately through the internal output-directory setting.

## 7. Linear question models

`models.question_models` defines:

```python
SUPPORTED_DIFFICULTIES = ("Easy", "Medium", "Hard")
SUPPORTED_LINEAR_QUESTION_TYPES = ("x_intercept", "y_intercept", "gradient")
```

`QuestionBlueprint` fields:

```text
subject: str = "Mathematics"
grade: int = 9
topic: str = "Functions"
subtopic: str = "Linear Functions"
difficulty: str = "Medium"
marks_per_question: int = 4
number_of_questions: int = 10
question_types: list[str] = [x_intercept, y_intercept, gradient]
```

`QuestionBlueprint.validate()` checks positive grade, marks, and count; supported difficulty; non-empty question types; and membership in the three supported Linear question types.

`LinearQuestionData` is frozen:

```text
equation: str
gradient: int | float
y_intercept: int | float
x_intercept: int | float | None
function_name: str = "f"
```

`GeneratedQuestion` fields:

```text
question_id, question_type, subject, grade, topic, subtopic, difficulty,
marks, question_text, expected_answer, memo, mathematical_data,
graph_request, graph_artifact
```

`QuestionBatch` fields:

```text
blueprint: QuestionBlueprint
questions: list[GeneratedQuestion]
batch_id: str
created_at: str (UTC ISO timestamp by default)
```

### Public Linear interfaces

```python
def build_linear_display_settings(question_type: str) -> GraphDisplaySettings

def build_question_text(question_type: str, equation: str | None = None) -> str

def build_memo(question_type: str, data: LinearQuestionData) -> tuple[str, str]

def generate_linear_question_batch(
    blueprint: QuestionBlueprint,
    *,
    seed: int | None = None,
    use_ai: bool = False,
) -> QuestionBatch
```

### Generation flow

```text
QuestionBlueprint
  -> validate blueprint
  -> choose question type with seeded random.Random
  -> choose gradient/y-intercept/x-intercept parameters
  -> build verified equation and answer
  -> build learner graph display policy
  -> build deterministic wording/memo
  -> optionally replace wording/memo via AI
  -> create GraphRequest
  -> generate PNG in batch directory
  -> create GeneratedQuestion
  -> return QuestionBatch
```

The current implementation creates the batch UUID and directory before entering the question loop. It constructs Python’s verified answer and deterministic fallback before any optional AI call. The graph is generated after the wording/memo decision. AI cannot replace `expected_answer` or graph settings.

## 8. Linear mathematics

`_candidate_parameters(difficulty)` builds `(gradient, y_intercept, x_intercept)` candidates. For Easy, gradients are `[-2, -1, 1, 2]` and roots range from -5 to 5. Medium uses nonzero gradients -5 through 5 and roots -5 through 5. Hard uses the listed nonzero gradients and roots -8 through 8. Candidates are retained only where `-10 <= y_intercept <= 10`.

`_generate_data(rng, difficulty)` chooses one candidate and creates `LinearQuestionData`:

```text
equation = m*x + c, formatted with x/-x and +/- c simplifications
y_intercept = c
x_intercept = -c / m
```

`_format_linear_equation()` formats gradient 1 as `x`, -1 as `-x`, and other gradients as `m*x`.

The random generator is local (`random.Random(seed)`), so seeded mathematical data is reproducible. Duplicate fingerprints are `(question_type, gradient, y_intercept)`. A duplicate is skipped. The loop allows `number_of_questions * 5` attempts and raises `ValueError` if the requested unique count cannot be produced.

Expected answers are Python-generated:

- `x_intercept`: `(x_intercept, 0)`;
- `y_intercept`: `(0, y_intercept)`;
- `gradient`: the formatted gradient.

No AI output is used for these values.

## 9. Linear learner-facing display policy

`build_linear_display_settings()` starts with `GraphDisplaySettings` defaults, then sets:

```text
show_title = False
show_legend = False
show_equation = False
```

Thus the generated PNG has no graph heading, legend, or equation label. This is separate from `app.py`, whose manual controls still default to and pass through user-selected title/legend/equation behavior.

For `x_intercept`:

- equation is included in deterministic question text as `The graph of f(x) = ...`;
- x-intercept marker/label is hidden;
- y-intercept is shown as reference;
- title, legend, and graph equation are hidden;
- gradient annotation and triangle retain their normal defaults, but no legend means they are not exposed through a legend.

For `y_intercept`:

- equation is included in question text;
- y-intercept marker/label is hidden;
- x-intercept is shown as reference;
- title, legend, and graph equation are hidden.

For `gradient`:

- equation is included in question text;
- x- and y-intercepts are shown as reference points;
- `show_gradient=False` and `show_gradient_triangle=False` prevent the answer from being displayed as graph annotations;
- title, legend, and graph equation are hidden.

`determine_equation` exists as scaffolding in `build_linear_display_settings()` and `build_question_text()`, with all equation/title/legend display disabled. It is not an allowed `QuestionBlueprint.question_types` value, so it cannot currently be generated through `generate_linear_question_batch()`.

## 10. AI architecture

### `ai/client.py`

The module imports the official SDK:

```python
from openai import OpenAI
```

`get_client()` returns `OpenAI()` lazily. There is no client construction or API request during module import. The SDK reads `OPENAI_API_KEY` from the environment.

`get_model()` returns `OPENAI_MODEL` when set, otherwise `gpt-4o-mini`.

`create_structured_response(*, system_prompt, user_prompt)` makes one Responses API call with the selected model and strict JSON schema. The schema requires exactly two string properties: `question_text` and `memo`; `additionalProperties` is false.

Environment variables:

```powershell
$env:OPENAI_API_KEY="..."   # required only for live AI requests; never commit it
$env:OPENAI_MODEL="gpt-4o-mini"  # optional override
$env:USE_AI="true"           # enables AI in run_linear_questions.py
$env:RUN_OPENAI_SMOKE="true" # enables test_openai.py
```

### `ai/question_writer.py`

Typed result:

```python
@dataclass(frozen=True)
class AIQuestionText:
    question_text: str
    memo: str
```

Public writer signature:

```python
def write_linear_question(
    *,
    grade: int,
    difficulty: str,
    question_type: str,
    equation: str,
    expected_answer: str,
    gradient: int | float,
    x_intercept: int | float | None,
    y_intercept: int | float,
    visible_information: list[str],
    hidden_information: list[str],
) -> AIQuestionText
```

The prompt identifies South African school Mathematics, grade/difficulty, verified values, visible/hidden information, the graph’s lack of title/legend/equation, the need for age-appropriate wording, and the requirement that the memo end with the answer. The user payload is JSON containing all supplied values and output requirements.

Current local validation:

- response must be a dict with exactly `question_text` and `memo`;
- both fields must be non-empty strings;
- either field containing case-insensitive `TODO` is rejected;
- `normalize_answer()` lowercases, removes spaces, semicolons are normalized to commas, and periods are removed;
- the normalized verified answer must occur in the normalized memo. This is the current implementation of the “memo must include/end with verified answer” rule: it enforces inclusion, not a strict final-character suffix;
- the question text must not contain the verified answer for x/y-intercept questions; gradient answer detection is intentionally skipped because the requested type necessarily names gradient and the current validator does not attempt to distinguish ordinary wording from a numeric reveal;
- if equation is visible, the display-equation form must occur in question text;
- the requested type phrase must occur, with underscores normalized to hyphens;
- a hidden gradient is rejected if `gradient:` appears in question text.

The writer makes at most two calls: the first attempt plus one retry after any exception, malformed JSON, or validation failure. It raises `ValueError` after the retry fails.

### Integration and fallback

`generate_linear_question_batch(..., use_ai=False)` never calls the AI writer and uses deterministic `build_question_text()` and `build_memo()`.

With `use_ai=True`, each question sends one writer request normally. The integration catches any exception from the writer, logs a warning such as `AI request failed for linear_0004; using deterministic fallback`, logs exception type/message, and continues the batch using the deterministic text/memo. Therefore one failed question does not destroy the batch. A successful AI result replaces only `question_text` and `memo`.

The current runner enables AI through `$env:USE_AI="true"`; it is disabled by default.

## 11. AI authority boundary

AI is allowed to control only:

```text
question_text
memo
```

Python remains the source of truth for:

```text
question type
equation
gradient
x/y intercepts
expected_answer
question uniqueness
graph display settings
graph range
PNG generation and path
QuestionBatch structure
JSON persistence
```

## 12. Batch-specific graph storage

A generated Linear batch uses:

```text
generated_graphs/linear_<uuid>/linear_0001.png
generated_graphs/linear_<uuid>/linear_0002.png
...
```

`generate_linear_question_batch()` creates `batch_id = "linear_" + uuid4().hex` and calls `mkdir(parents=True, exist_ok=False)` before generating questions. Each request keeps a safe basename such as `linear_0001.png`; `generate_graph_from_request()` receives the batch directory separately. The resulting artifact stores the exact nested path string, and `asdict()` persistence stores that path in JSON.

This prevents two batches from overwriting one another even when both contain `linear_0001.png`. `questions.persistence.validate_batch_graph_files()` resolves every stored path through `batch_loader.resolve_graph_path()` and checks `is_file()` before saving by default.

## 13. Persistence

File: `questions/persistence.py`.

Public functions:

```python
def question_batch_to_dict(batch: QuestionBatch) -> dict[str, object]

def validate_batch_graph_files(
    batch: QuestionBatch,
) -> tuple[bool, list[str]]

def save_question_batch(
    batch: QuestionBatch,
    output_path: str | Path,
    *,
    validate_graph_files: bool = True,
) -> Path
```

Serialization uses `dataclasses.asdict()` followed by indented JSON. The JSON contains `blueprint`, `questions`, `batch_id`, and `created_at`; each question contains its model fields, nested mathematical data, graph request, and graph artifact.

`save_question_batch()` creates the output parent directory, validates graph files by default, raises `ValueError` listing missing paths, and writes the JSON only after validation. `validate_graph_files=False` is available for callers that intentionally want to save references to missing files.

## 14. Review UI and batch loader

`batch_loader.py` public functions:

```python
def get_batches_directory() -> Path
def discover_batch_files() -> list[Path]
def load_batch_from_file(file_path: Path) -> tuple[Optional[QuestionBatch], Optional[str]]
def load_batch(file_name: str) -> tuple[Optional[dict], Optional[str]]
def resolve_graph_path(image_path: str) -> Path
def get_available_question_types(batch_data: dict) -> list[str]
def filter_questions(batch_data: dict, question_type: Optional[str] = None) -> list[dict]
```

`discover_batch_files()` finds sorted `*.json` files under `generated_questions`. `load_batch()` returns raw dict data for the reviewer and validates basic presence of a non-empty question list. `load_batch_from_file()` creates a lightweight `QuestionBatch` and stores raw questions on `_raw_questions` for display.

`resolve_graph_path()` normalizes backslashes to forward slashes, splits path components, and resolves them under the project root. It supports both legacy flat paths and nested `generated_graphs/<batch_id>/...` paths.

`review_questions.py`:

- initializes session keys for current batch/name, question index, and filter;
- uses callbacks for Previous, Next, and question selection;
- clamps the current index after filtering to avoid out-of-range errors;
- discovers/selects a JSON batch in the sidebar;
- displays summary metadata and batch ID/creation time;
- filters by question type;
- displays learner question text and graph image;
- provides Answer and Memo expanders;
- provides Admin Details with mathematical data, graph request, and full request JSON;
- warns when a graph path is absent or missing;
- refreshes batches or page through buttons.

The navigation bug was addressed by making `current_question_index` authoritative in session state, using button callbacks to mutate it, and clamping it against the filtered question list during each render.

## 15. Manual graph UI distinction

`app.py` is a separate manual graph-generation interface. It exposes title, legend, equation, curve-label, intercept, point-label, axes, range, and graph-specific controls, then constructs `GraphSettings` and calls `generate_graph()`.

Question-generated graphs use `GraphRequest` display settings built by `questions.linear.build_linear_display_settings()`, where title, legend, and graph equation are disabled. This does not alter manual `app.py` settings or standalone graph API defaults.

## 16. Tests

Current test files and coverage:

| File | Coverage |
|---|---|
| `tests/test_ai_question_writer.py` | AI disabled no-call behavior, mocked AI replacement, answer/display preservation, exception fallback, seeded math invariance, lazy import, and one retry. |
| `tests/test_batch_loader.py` | Batch discovery/loading, path normalization, nested graph path resolution, filtering, and question-type extraction. |
| `tests/test_cartesian_axes.py` | Central Cartesian axes, ticks, labels, arrows, and origin handling. |
| `tests/test_display_controls.py` | Curve labels, intercept label styles, point registry, and display toggles. |
| `tests/test_graph_api.py` | Supported type registry, graph artifact creation, mixed validation, and unsupported types. |
| `tests/test_graph_request.py` | Request conversion/validation, ranges, equations, display mapping, and unsafe output names. |
| `tests/test_linear_display_policy.py` | Learner title/legend/equation policy, intercept hiding, gradient hiding, future `determine_equation` scaffolding, and memo behavior. |
| `tests/test_linear_questions.py` | Blueprint validation, deterministic mathematics, batch isolation, question graph settings/text, PNG creation, and JSON persistence. |
| `tests/test_mixed.py` | Mixed type detection, roots/intersections, asymptotes, and supported pair rendering. |
| `tests/test_parsing.py` | Shared parser and Linear/Quadratic `y = ...` input handling. |
| `tests/test_persistence.py` | Missing-file validation, save behavior, optional validation, and batch serialization. |
| `test_openai.py` | Live smoke test, skipped unless `RUN_OPENAI_SMOKE=true`; no import-time API call. |

Latest full offline/default run:

```text
105 passed, 1 skipped, 21 warnings
```

The skip is the opt-in live OpenAI smoke test. The warnings are primarily non-interactive Matplotlib warnings and expected Mixed asymptote warnings.

## 17. Environment and commands

Assumptions:

- Windows;
- Python 3.11 is the intended documented interpreter;
- dependencies are listed in `requirements.txt`: Matplotlib, NumPy, SymPy, Pillow, Streamlit, OpenAI;
- Python 3.14 was used in recent verification and worked for offline tests after installing dependencies, but Python 3.11 remains the project command convention.

Commands:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 run_linear_questions.py
py -3.11 -m streamlit run app.py
py -3.11 -m streamlit run review_questions.py
py -3.11 -m pytest
```

AI configuration:

```powershell
$env:USE_AI="true"
$env:OPENAI_API_KEY="<set securely in the shell>"
$env:OPENAI_MODEL="gpt-4o-mini"
py -3.11 run_linear_questions.py
```

`USE_AI` defaults false. `OPENAI_MODEL` defaults to `gpt-4o-mini`. The SDK reads `OPENAI_API_KEY`; no key is stored in source. `RUN_OPENAI_SMOKE` defaults false.

## 18. Matplotlib backend and warnings

`run_linear_questions.py` sets `matplotlib.use("Agg")` before importing graph-generation modules, so batch generation does not open blocking GUI windows. Several generator modules still call `plt.show()` after saving. Under the Agg backend this produces warnings such as:

```text
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
```

The current cleanup location is the final `plt.show()` call in each generator, especially `generators/linear.py` near the end of `create_linear_graph()`. Removing or guarding those calls would reduce warnings, but is outside the current AI phase.

## 19. Known limitations and unfinished work

- AI output validation is local and heuristic. It validates required fields, placeholders, answer inclusion in the memo, visible equation presence, question type phrase, and one hidden-gradient pattern, but it is not a complete semantic checker.
- The current memo rule requires the normalized verified answer to be present in the memo; despite prompt wording, it does not strictly verify that the memo ends with the answer.
- Gradient question text necessarily names “gradient”; numeric answer leakage detection is intentionally limited.
- AI integration catches broad exceptions and logs the exception type/message. Future logging should be reviewed to ensure remote error text cannot contain sensitive data.
- Live OpenAI validation depends on network access, credentials, model availability, and SDK compatibility. It is not part of the default test run.
- The live smoke test remains a network test and is skipped by default.
- `plt.show()` calls remain and produce Agg warnings.
- Generated PNG directories accumulate; there is no retention or cleanup policy.
- `generated_questions/linear_test_batch.json` is overwritten by the runner, although its referenced graph files live in immutable UUID directories.
- Mixed graphs support only Linear, Quadratic, supported Exponential, and Hyperbola members.
- Only three Linear question types are currently enabled: `x_intercept`, `y_intercept`, and `gradient`.
- `determine_equation` is display/text scaffolding only and is rejected by `QuestionBlueprint.validate()`.
- `questions.persistence` has a placeholder/pass in one existing test that loads the canonical batch but does not construct a full validation assertion.
- The old project has accumulated untracked helper/test/UI files in the current workspace; inspect `git status --short` before committing and do not assume every untracked file belongs to a single change.

## 20. Current implementation status

| Area | Status | Notes |
|---|---|---|
| Individual graph generators | Implemented | Eleven supported graph types through graph API. |
| Graph API | Implemented | Request validation, settings conversion, artifact existence check. |
| Linear mathematics | Implemented | Seeded deterministic values and duplicate prevention. |
| Linear learner graph policy | Implemented | No title, legend, or graph equation; answer-specific hiding. |
| Batch graph isolation | Implemented | UUID directory per Linear batch. |
| JSON persistence | Implemented | Dataclass serialization and default graph validation. |
| Review UI | Implemented | Read-only display with nested path support and callbacks. |
| AI Phase 1 | Implemented with fallback | Optional wording/memo generation only; offline by default. |
| Live AI smoke | Opt-in | Requires network and `OPENAI_API_KEY`. |
| Additional Linear question types | Not implemented | Scaffolding exists only for `determine_equation`. |
| Quadratic question generation | Not implemented | Quadratic graph generation exists. |
| XtraClass/database integration | Not implemented | No current integration code. |

## 21. Explicit roadmap classification

Implemented:

- deterministic Linear question generation;
- verified Python answers;
- graph display policy;
- batch-specific graph storage;
- JSON persistence and validation;
- Streamlit graph and review interfaces;
- optional AI wording/memo Phase 1;
- offline tests and opt-in live smoke test.

Planned or explicitly suggested by the current project/request history:

- stabilize AI wording and validation;
- add Linear types such as `read_coordinate`, `find_f_of_x`, and `determine_equation`;
- add approve/reject/regenerate workflow;
- build a Quadratic question generator;
- add more graph/visual categories;
- eventual XtraClass persistence integration.

Suggested engineering follow-ups, not currently implemented commitments:

- register an integration pytest marker or keep the current skip-only smoke style;
- add a strict memo suffix validator if that contract is required;
- avoid logging raw exception messages from remote providers;
- remove/guard `plt.show()` for batch mode;
- add generated-artifact retention/cleanup;
- add a full end-to-end persistence test that loads the regenerated JSON and validates all nested paths;
- consider separating generated output roots from source-controlled fixture images.

## 22. WHERE TO CHANGE THINGS

| Goal | Files/functions to inspect |
|---|---|
| Change Linear mathematical values | `questions/linear.py`: `_candidate_parameters()`, `_generate_data()`, `_format_linear_equation()` |
| Change Linear expected answers | `questions/linear.py`: `build_memo()` and the answer construction inside `generate_linear_question_batch()` |
| Change deterministic question wording | `questions/linear.py`: `build_question_text()` |
| Change deterministic memo wording | `questions/linear.py`: `build_memo()` |
| Change AI prompt | `ai/question_writer.py`: `_prompts()` |
| Change AI model | `ai/client.py`: `DEFAULT_MODEL`, `get_model()`; environment `OPENAI_MODEL` |
| Change AI structured output | `ai/client.py`: `create_structured_response()` schema |
| Change AI validation | `ai/question_writer.py`: `normalize_answer()`, `_validate_response()` |
| Change AI fallback/retry/logging | `ai/question_writer.py`: `write_linear_question()`; `questions/linear.py`: `use_ai` block |
| Enable/disable AI in runner | `run_linear_questions.py`: `USE_AI` environment lookup |
| Change Linear learner graph policy | `questions/linear.py`: `build_linear_display_settings()` |
| Change title/legend/equation rendering globally | `models/graph_request.py`, `models/graph_settings.py`, `generators/api.py`, `generators/graph_helpers.py` |
| Change Linear graph rendering | `generators/linear.py`: `create_linear_graph()` |
| Change graph output path handling | `generators/api.py`: `graph_request_to_settings()`, `generate_graph()`; graph-specific generator output-directory assignment |
| Change batch directory naming | `questions/linear.py`: `batch_id` and `batch_output_directory` creation |
| Change JSON shape | `questions/persistence.py`: `question_batch_to_dict()`, model dataclasses |
| Change graph-file validation | `questions/persistence.py`: `validate_batch_graph_files()`; `batch_loader.resolve_graph_path()` |
| Change reviewer batch discovery/loading | `batch_loader.py`: `discover_batch_files()`, `load_batch()`, `resolve_graph_path()` |
| Change reviewer display | `review_questions.py`: `display_batch_summary()`, `display_question()` |
| Change reviewer navigation | `review_questions.py`: `handle_previous_click()`, `handle_next_click()`, `handle_question_select()`, `display_navigation()`, `main()` |
| Add a Linear question type | `models/question_models.py` supported-type tuple and validation, `questions/linear.py` policy/text/memo/generation, AI prompts/validation, and tests |
| Add a graph type | new `generators/<type>.py`, `constants.py`, `generators/api.py`, request/settings mapping as needed, and graph/API tests |
| Change parsing | `generators/graph_helpers.py` and graph-type-specific parser modules |
| Change manual UI | `app.py`; do not change question policy unless intended |
| Change console manual UI | `main.py` |

## 23. Dependency map

```text
run_linear_questions.py
  -> models.question_models.QuestionBlueprint
  -> questions.linear.generate_linear_question_batch
  -> questions.persistence.save_question_batch

questions.linear
  -> models.question_models
  -> models.graph_request
  -> ai.question_writer (imported, called only when use_ai=True)
  -> generators.api.generate_graph_from_request

ai.question_writer
  -> ai.client.create_structured_response
  -> OpenAI Responses API

questions.persistence
  -> batch_loader.resolve_graph_path
  -> dataclasses.asdict / JSON

review_questions.py
  -> batch_loader
  -> Streamlit

app.py
  -> generators.api.generate_graph
  -> graph-specific parsers/helpers

generators.api
  -> GraphRequest
  -> GraphSettings
  -> graph-specific generator modules
  -> GraphArtifact

graph-specific generators
  -> generators.graph_helpers
  -> SymPy, NumPy, Matplotlib
```

## 24. Important interface snippets

```python
# generators/api.py
def generate_graph(
    graph_type: str,
    settings: GraphSettings,
    equation: str | None = None,
    equations: list[str] | None = None,
) -> GraphArtifact

def generate_graph_from_request(
    request: GraphRequest,
    *,
    output_directory: Path | str | None = None,
) -> GraphArtifact

def graph_request_to_settings(
    request: GraphRequest,
    *,
    output_directory: Path | str | None = None,
) -> GraphSettings
```

```python
# questions/linear.py
def generate_linear_question_batch(
    blueprint: QuestionBlueprint,
    *,
    seed: int | None = None,
    use_ai: bool = False,
) -> QuestionBatch
```

```python
# ai/question_writer.py
def write_linear_question(
    *,
    grade: int,
    difficulty: str,
    question_type: str,
    equation: str,
    expected_answer: str,
    gradient: int | float,
    x_intercept: int | float | None,
    y_intercept: int | float,
    visible_information: list[str],
    hidden_information: list[str],
) -> AIQuestionText
```

```python
# questions/persistence.py
def save_question_batch(
    batch: QuestionBatch,
    output_path: str | Path,
    *,
    validate_graph_files: bool = True,
) -> Path

def validate_batch_graph_files(
    batch: QuestionBatch,
) -> tuple[bool, list[str]]
```

## 25. Final inspected workspace state

`git status --short` at handover creation reported these modified tracked files:

```text
generated_questions/linear_test_batch.json
generators/api.py
generators/circle.py
generators/cubic.py
generators/exponential.py
generators/hyperbola.py
generators/linear.py
generators/logarithmic.py
generators/mixed.py
generators/quadratic.py
generators/tangent.py
generators/trig_helpers.py
models/graph_request.py
models/graph_settings.py
questions/linear.py
questions/persistence.py
requirements.txt
run_linear_questions.py
tests/test_linear_questions.py
```

Untracked paths reported included:

```text
ai/
batch_loader.py
review_questions.py
test_openai.py
tests/test_ai_question_writer.py
tests/test_batch_loader.py
tests/test_linear_display_policy.py
tests/test_persistence.py
generated_graphs/linear_204d00f323b64da1bc71c3cf15db9470/
generated_graphs/linear_95f9863682354a8d8f32e5dd82ad2e39/
generated_graphs/linear_a1430668fd904c5cb1b8b3ad3f7e44b8/
generated_graphs/linear_d55065bbdee348ef8dd946c01c2e49cf/
```

The exact generated-directory set can change after tests or batch generation. No files were deleted or reverted for this report. Review the actual `git status --short` before committing.

## 26. Short continuation checklist

1. Read this report and inspect the current versions of `questions/linear.py` and `ai/question_writer.py`, because those files have recently received manual/automated edits.
2. Run `py -3.11 -m pytest` for the default offline suite.
3. Keep `USE_AI` false while developing deterministic changes.
4. Mock `ai.question_writer.write_linear_question` in unit tests; never add ordinary tests that call OpenAI.
5. Treat `expected_answer`, `LinearQuestionData`, `GraphRequest`, and graph artifact paths as Python-owned data.
6. When changing output paths, preserve nested batch directories and update both persistence and reviewer resolver tests.
7. Use `RUN_OPENAI_SMOKE=true` only for an intentional live network test.
