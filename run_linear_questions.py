import matplotlib
import os

# Use a non-interactive Matplotlib backend so graph windows
# do not open and block question generation.
matplotlib.use("Agg")

from models.question_models import QuestionBlueprint
from questions.linear import generate_linear_question_batch
from questions.persistence import save_question_batch


def main():
    blueprint = QuestionBlueprint(
        grade=9,
        difficulty="Medium",
        number_of_questions=3,
        question_types=[
            "determine_equation",
        ],
    )

    batch = generate_linear_question_batch(
        blueprint,
        seed=123,
        use_ai=os.environ.get("USE_AI", "false").lower() == "true",
    )

    try:
        output_path = save_question_batch(
            batch,
            "generated_questions/linear_test_batch.json",
            validate_graph_files=True,
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Generated {len(batch.questions)} questions but could not save batch due to missing graph files.")
        raise

    print(f"Generated {len(batch.questions)} questions")
    print(f"Saved batch to: {output_path}")

    for question in batch.questions:
        print()
        print("----------------------------")
        print("ID:", question.question_id)
        print("Type:", question.question_type)
        print("Question:", question.question_text)
        print("Answer:", question.expected_answer)
        print("Graph:", question.graph_artifact.image_path)


if __name__ == "__main__":
    main()