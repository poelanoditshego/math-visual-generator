# Question engine architecture

All generation modes produce a `QuestionBlueprint`. The generic engine validates that
blueprint, selects a registered `QuestionSpec`, coordinates batch IDs and graph files,
and builds the final `GeneratedQuestion` records. Persistence and Review Questions keep
consuming the same `QuestionBatch` JSON structure.

`questions/specs.py` is the registry for learner-facing configuration: default wording,
answer category, graph display overrides, uniqueness fields, and the information AI may
or may not see. It contains no mathematical formulas.

`questions/linear.py` owns the Linear family. It chooses coefficients and points,
calculates authoritative answers, builds deterministic memos, selects graph ranges, and
creates Linear graph requests. Its public helpers and `generate_linear_question_batch`
remain compatibility entry points; the latter delegates orchestration to
`questions/engine.py`.

Python remains authoritative for equations, values, answers, graph parameters, rendering,
and persistence. AI may classify source input or rewrite wording and a memo. A failed or
invalid AI response falls back to the deterministic Python content without changing the
expected answer.

## Input flow

By topic, example question, transcript, and image inputs each build the same blueprint.
`generate_questions.generate_and_save_question_batch` sends it through the generic engine
and the common JSON persistence path, then the UI records `latest_generated_batch`.

## Add a Linear question type

1. Add its stable name to `SUPPORTED_LINEAR_QUESTION_TYPES` and register one
   `QuestionSpec`, including display and AI visibility policies.
2. Add only the required Linear math/data handling and deterministic answer/memo in
   `questions/linear.py`. Reuse the existing path when no new math is needed.
3. Add focused registry, math, leakage, and display tests.

Do not put formulas in the engine or make AI responsible for expected answers.

`intersection_of_two_lines` demonstrates the intended extension path: its specification
defines display and AI policy, the Linear family constructs two equations and their exact
intersection, and the unchanged Mixed graph renderer draws them from a standard
`GraphRequest(graph_type="Mixed", equations=[...])`.

## Add a future family

Create its mathematical-data model and an implementation of `QuestionFamilyGenerator`,
register its specifications and generator, and set the blueprint's `family`. The family
produces candidates; the engine continues to own batch orchestration and artifacts. Keep
family-specific formulas and validation out of the engine.
