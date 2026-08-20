from models.question_models import QuestionBlueprint
from questions.linear import generate_linear_question_batch
from questions.persistence import save_question_batch


blueprint = QuestionBlueprint(
    grade=9,
    difficulty="Medium",
    number_of_questions=10,
)

batch = generate_linear_question_batch(
    blueprint,
    seed=123,
)

output_path = save_question_batch(
    batch,
    "generated_questions/linear_test_batch.json",
)

print(f"Generated {len(batch.questions)} questions")
print(f"Saved batch to: {output_path}")

for question in batch.questions:
    print()
    print(question.question_id)
    print(question.question_type)
    print(question.question_text)
    print("Answer:", question.expected_answer)
    print("Graph:", question.graph_artifact.image_path)