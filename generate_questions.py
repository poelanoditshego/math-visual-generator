from pathlib import Path

import streamlit as st

from ai.example_question_analyzer import (
    ExampleQuestionAnalysis,
    analyze_example_question,
)
from ai.image_question_analyzer import (
    ImageQuestionAnalysis,
    analyze_image_question,
)
from ai.transcript_analyzer import TranscriptAnalysis, analyze_transcript
from models.question_models import QuestionBlueprint
from questions.blueprints import (
    build_example_question_blueprint,
    build_image_question_blueprint,
    build_topic_question_blueprint,
    build_transcript_question_blueprint,
)
from questions.engine import generate_question_batch
from questions.persistence import save_question_batch



QUESTION_TYPES = {
    "X-intercept": "x_intercept",
    "Y-intercept": "y_intercept",
    "Gradient": "gradient",
    "Determine equation": "determine_equation",
    "Equation from two points": "equation_from_two_points",
    "Equation from gradient and point": "equation_from_gradient_and_point",
    "Find f(x)": "find_f_of_x",
    "Find x given y": "find_x_given_y",
    "Read coordinate": "read_coordinate",
    "Increasing or decreasing": "increasing_or_decreasing",
    "Intersection of two lines": "intersection_of_two_lines",
    "Parallel lines": "parallel_lines",
    "Perpendicular lines": "perpendicular_lines",
}
QUESTION_TYPE_LABELS = {internal: label for label, internal in QUESTION_TYPES.items()}


def generate_and_save_question_batch(
    blueprint: QuestionBlueprint,
    *,
    use_ai: bool = False,
):
    """Run the shared engine/persistence pipeline used by all four input modes."""
    batch = generate_question_batch(blueprint, use_ai=use_ai)
    output_path = Path("generated_questions") / f"{batch.batch_id}.json"
    save_question_batch(batch, output_path)
    return batch, output_path


def show_question_generator() -> None:
    """Render the question-generation page within the unified application."""

    st.title("Mathematics Question Generator")
    st.write("Select how you want to generate questions.")

    generation_method = st.radio(
        "How do you want to generate questions?",
        options=[
            "By topic",
            "From an example question",
            "From a transcript",
            "From an image",
        ],
        horizontal=True,
    )

    if generation_method == "By topic":
        grade = st.selectbox(
            "Grade",
            options=[8, 9, 10, 11, 12],
            index=1,
        )

        difficulty = st.selectbox(
            "Difficulty",
            options=[
                "Easy",
                "Medium",
                "Hard",
            ],
        )

        number_of_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        selected_labels = st.multiselect(
            "Question types",
            options=list(QUESTION_TYPES.keys()),
            default=[],
            placeholder="Select one or more question types",
        )

        use_ai = st.checkbox(
            "Use AI for question wording and memo",
            value=False,
        )

        if st.button(
            "Generate Questions",
            type="primary",
            use_container_width=True,
        ):
            if not selected_labels:
                st.error("Please select at least one question type.")
                return

            blueprint = build_topic_question_blueprint(
                [QUESTION_TYPES[label] for label in selected_labels],
                grade=int(grade),
                difficulty=difficulty,
                number_of_questions=int(number_of_questions),
            )

            try:
                with st.spinner(
                    "Generating questions and graphs..."
                ):
                    batch, output_path = generate_and_save_question_batch(
                        blueprint,
                        use_ai=use_ai,
                    )

                st.session_state[
                    "latest_generated_batch"
                ] = output_path.name

                st.success(
                    f"Successfully generated "
                    f"{len(batch.questions)} questions."
                )

                st.write(
                    f"Batch ID: `{batch.batch_id}`"
                )

                st.write(
                    f"Saved to: `{output_path}`"
                )

            except Exception as error:
                st.error(
                    f"Question generation failed: {error}"
                )

    elif generation_method == "From an example question":
        st.subheader(
            "Generate from an example question"
        )

        example_question = st.text_area(
            "Example question",
            placeholder=(
                "Example: Determine the x-intercept of "
                "f(x) = 2x - 6."
            ),
            height=140,
        )

        grade = st.selectbox(
            "Grade",
            options=[8, 9, 10, 11, 12],
            index=1,
            key="example_grade",
        )

        number_of_questions = st.number_input(
            "Number of similar questions",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="example_question_count",
        )

        difficulty = st.selectbox(
            "Difficulty",
            options=["Easy", "Medium", "Hard"],
            key="example_difficulty",
        )

        use_ai = st.checkbox(
            "Use AI for question wording and memo",
            value=False,
            key="example_use_ai",
        )

        st.info(
            "The system will analyse the example and "
            "generate similar questions with different "
            "mathematical values."
        )

        if st.button(
            "Generate Similar Questions",
            use_container_width=True,
        ):
            if not example_question.strip():
                st.error("Please provide an example question.")
                return
            try:
                with st.spinner("Analysing the example and generating questions..."):
                    analysis = analyze_example_question(example_question)
                    blueprint = build_example_question_blueprint(
                        analysis,
                        grade=int(grade),
                        difficulty=difficulty,
                        number_of_questions=int(number_of_questions),
                    )
                    batch, output_path = generate_and_save_question_batch(
                        blueprint, use_ai=use_ai
                    )
                st.session_state["latest_generated_batch"] = output_path.name
                st.success(f"Successfully generated {len(batch.questions)} similar questions.")
                st.write(f"Detected type: `{analysis.question_type}`")
                st.write(f"Batch ID: `{batch.batch_id}`")
                st.write(f"Saved to: `{output_path}`")
            except ValueError as error:
                st.error(f"Example-question generation failed: {error}")
            except Exception as error:
                st.error(f"Example-question generation failed: {error}")

    elif generation_method == "From a transcript":
        st.subheader(
            "Generate from a lesson transcript"
        )

        transcript = st.text_area(
            "Video or lesson transcript",
            placeholder="Paste the transcript here...",
            height=300,
        )

        grade = st.selectbox(
            "Grade",
            options=[8, 9, 10, 11, 12],
            index=1,
            key="transcript_grade",
        )

        number_of_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="transcript_question_count",
        )

        difficulty = st.selectbox(
            "Difficulty",
            options=["Easy", "Medium", "Hard"],
            key="transcript_difficulty",
        )

        use_ai = st.checkbox(
            "Use AI for question wording and memo",
            value=False,
            key="transcript_use_ai",
        )

        st.info(
            "The system will identify the concepts "
            "covered in the transcript and generate "
            "relevant questions."
        )

        if st.button(
            "Generate from Transcript",
            use_container_width=True,
        ):
            if not transcript.strip():
                st.error("Please provide a transcript.")
                return
            try:
                with st.spinner("Analysing the transcript and generating questions..."):
                    analysis = analyze_transcript(transcript)
                    blueprint = build_transcript_question_blueprint(
                        analysis,
                        grade=int(grade),
                        difficulty=difficulty,
                        number_of_questions=int(number_of_questions),
                    )
                    batch, output_path = generate_and_save_question_batch(
                        blueprint, use_ai=use_ai
                    )
                st.session_state["latest_generated_batch"] = output_path.name
                st.success(f"Successfully generated {len(batch.questions)} questions.")
                st.write("Detected question types:")
                for question_type in analysis.question_types:
                    st.write(f"- {QUESTION_TYPE_LABELS[question_type]}")
                st.write(f"Batch ID: `{batch.batch_id}`")
                st.write(f"Saved to: `{output_path}`")
            except ValueError as error:
                st.error(f"Transcript generation failed: {error}")
            except Exception as error:
                st.error(f"Transcript generation failed: {error}")

    elif generation_method == "From an image":
        st.subheader(
            "Generate from a question image"
        )

        uploaded_image = st.file_uploader(
            "Upload an example question",
            type=[
                "png",
                "jpg",
                "jpeg",
            ],
        )

        if uploaded_image is not None:
            st.image(
                uploaded_image,
                caption="Uploaded example",
                width=500,
            )

        grade = st.selectbox(
            "Grade",
            options=[8, 9, 10, 11, 12],
            index=1,
            key="image_grade",
        )

        number_of_questions = st.number_input(
            "Number of similar questions",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="image_question_count",
        )

        difficulty = st.selectbox(
            "Difficulty",
            options=["Easy", "Medium", "Hard"],
            key="image_difficulty",
        )

        use_ai = st.checkbox(
            "Use AI for question wording and memo",
            value=False,
            key="image_use_ai",
        )

        st.info(
            "The system will analyse the question and "
            "graph, then generate similar questions "
            "with new values."
        )

        if st.button(
            "Generate from Image",
            type="primary",
            use_container_width=True,
        ):
            if uploaded_image is None:
                st.error("Please upload an image.")
                return

            image_bytes = uploaded_image.getvalue()
            if not image_bytes:
                st.error("Uploaded image file is empty.")
                return

            media_type = uploaded_image.type or f"image/{uploaded_image.name.rsplit('.', 1)[-1].lower()}"

            try:
                with st.spinner("Analysing the image and generating questions..."):
                    analysis = analyze_image_question(image_bytes, media_type=media_type)
                    blueprint = build_image_question_blueprint(
                        analysis,
                        grade=int(grade),
                        difficulty=difficulty,
                        number_of_questions=int(number_of_questions),
                    )
                    batch, output_path = generate_and_save_question_batch(
                        blueprint, use_ai=use_ai
                    )
                st.session_state["latest_generated_batch"] = output_path.name
                st.success(f"Successfully generated {len(batch.questions)} questions.")
                st.write("Detected question types:")
                for question_type in analysis.question_types:
                    st.write(f"- {QUESTION_TYPE_LABELS[question_type]}")
                st.write(f"Batch ID: `{batch.batch_id}`")
                st.write(f"Saved to: `{output_path}`")
            except ValueError as error:
                st.error(f"Image question generation failed: {error}")
            except Exception as error:
                st.error(f"Image question generation failed: {error}")



if __name__ == "__main__":
    st.set_page_config(
        page_title="Question Generator",
        layout="wide",
    )

    show_question_generator()
