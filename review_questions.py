"""Streamlit interface for reviewing generated question batches."""

import streamlit as st
from pathlib import Path
import json

from batch_loader import (
    discover_batch_files,
    load_batch,
    resolve_graph_path,
    get_available_question_types,
    filter_questions,
)


def init_session_state():
    """Initialize session state variables."""
    if "current_batch" not in st.session_state:
        st.session_state.current_batch = None
    if "current_batch_name" not in st.session_state:
        st.session_state.current_batch_name = None
    # AUTHORITATIVE session state key for current question index
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "current_filter" not in st.session_state:
        st.session_state.current_filter = "All"


def handle_previous_click():
    """Callback for Previous button: decrement index."""
    if st.session_state.current_question_index > 0:
        st.session_state.current_question_index -= 1


def handle_next_click():
    """Callback for Next button: increment index."""
    # We'll validate the upper bound in main() since we don't know filtered length here
    st.session_state.current_question_index += 1


def handle_question_select(new_index: int):
    """Callback for question selector: set index directly."""
    st.session_state.current_question_index = new_index


def display_batch_summary(batch_data: dict) -> None:
    """Display summary information about the batch."""
    blueprint = batch_data.get("blueprint", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Subject", blueprint.get("subject", "N/A"))
    with col2:
        st.metric("Grade", blueprint.get("grade", "N/A"))
    with col3:
        st.metric("Difficulty", blueprint.get("difficulty", "N/A"))
    with col4:
        st.metric("Questions", len(batch_data.get("questions", [])))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Topic:** {blueprint.get('topic', 'N/A')}")
    with col2:
        st.write(f"**Subtopic:** {blueprint.get('subtopic', 'N/A')}")
    with col3:
        st.write(f"**Marks/Q:** {blueprint.get('marks_per_question', 'N/A')}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Batch ID:** {batch_data.get('batch_id', 'N/A')}")
    with col2:
        st.write(f"**Created:** {batch_data.get('created_at', 'N/A')}")


def display_question(question: dict, question_index: int, total_questions: int) -> None:
    """Display a single question with all its details."""
    
    st.divider()
    
    # Question header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## Question {question_index + 1} of {total_questions}")
    with col2:
        st.markdown(f"**ID:** `{question.get('question_id', 'N/A')}`")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Type:** {question.get('question_type', 'N/A')}")
    with col2:
        st.write(f"**Marks:** {question.get('marks', 'N/A')}")
    with col3:
        st.write(f"**Difficulty:** {question.get('difficulty', 'N/A')}")
    
    st.divider()
    
    graph_role = question.get(
        "graph_role",
        "memo" if question.get("question_type") == "draw_linear_graph" else "question",
    )

    # Main question display (left) and graph (right)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### Question")
        st.markdown(f"**{question.get('question_text', 'N/A')}**")

        # Answer expander
        with st.expander("Show Answer"):
            st.code(question.get("expected_answer", "N/A"))

        # Memo expander
        with st.expander("Show Memo"):
            st.text(question.get("memo", "N/A"))
            if graph_role == "memo":
                st.markdown("#### Solution Graph")
                graph_artifact = question.get("graph_artifact", {})
                image_path_str = graph_artifact.get("image_path", "")
                if image_path_str:
                    image_path = resolve_graph_path(image_path_str)
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning(f"⚠️ Solution graph image not found:\n`{image_path}`")

    with right_col:
        if graph_role == "memo":
            st.markdown("### Solution Graph (Memo Only)")
            st.info(
                "ℹ️ This graph is for the memo/solution only and is NOT shown to the learner in the question."
            )
        else:
            st.markdown("### Graph")

        # Try to load and display the graph
        graph_artifact = question.get("graph_artifact", {})
        image_path_str = graph_artifact.get("image_path", "")

        if image_path_str:
            image_path = resolve_graph_path(image_path_str)

            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"⚠️ Graph image not found:\n`{image_path}`")
        else:
            st.warning("⚠️ No graph path found in question data")

    
    st.divider()
    
    # Admin details expander
    with st.expander("Admin Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mathematical Data")
            math_data = question.get("mathematical_data", {})
            st.write(f"**Equation:** `{math_data.get('equation', 'N/A')}`")
            st.write(f"**Gradient:** {math_data.get('gradient', 'N/A')}")
            st.write(f"**Y-intercept:** {math_data.get('y_intercept', 'N/A')}")
            if math_data.get('x_intercept') is not None:
                st.write(f"**X-intercept:** {math_data.get('x_intercept')}")
        
        with col2:
            st.subheader("Graph Details")
            graph_req = question.get("graph_request", {})
            graph_range = graph_req.get("graph_range", {})
            st.write(f"**Graph Type:** {graph_req.get('graph_type', 'N/A')}")
            st.write(f"**Output:** {graph_req.get('output_name', 'N/A')}")
            st.write(f"**Range:** X=[{graph_range.get('x_min', 'N/A')}, {graph_range.get('x_max', 'N/A')}]")
            st.write(f"**Range:** Y=[{graph_range.get('y_min', 'N/A')}, {graph_range.get('y_max', 'N/A')}]")
        
        st.subheader("Full GraphRequest JSON")
        st.json(question.get("graph_request", {}))


def display_navigation(current_index: int, total_questions: int) -> None:
    """
    Display navigation controls using session state callbacks.
    
    Uses session state callbacks to ensure buttons and selectbox work together properly.
    The selectbox doesn't directly update current_index; callbacks do.
    """
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    
    with col1:
        st.button(
            "⬅️ Previous",
            disabled=(current_index == 0),
            on_click=handle_previous_click,
            key="prev_button"
        )
    
    with col2:
        # Question selector - use a callback to update session state
        # This ensures the selectbox doesn't override button clicks
        st.selectbox(
            "Jump to question:",
            options=range(total_questions),
            format_func=lambda i: f"Question {i + 1} of {total_questions}",
            index=current_index,
            on_change=handle_question_select,
            args=(st.session_state.get("question_selector_value", current_index),),
            key="question_selector",
        )
        # Update the helper key with current selection for next callback
        if "question_selector" in st.session_state:
            st.session_state.question_selector_value = st.session_state.question_selector
    
    with col3:
        st.button(
            "Next ➡️",
            disabled=(current_index >= total_questions - 1),
            on_click=handle_next_click,
            key="next_button"
        )
    
    with col4:
        if st.button("🔄 Refresh", key="refresh_button"):
            st.rerun()


def show_question_reviewer() -> None:
    """Render the batch-review page within the unified application."""
    st.title("📋 Question Batch Reviewer")
    
    init_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Batch Selection")
        
        # Discover batches
        batch_files = discover_batch_files()
        
        if not batch_files:
            st.warning(
                "⚠️ No generated question batches were found.\n\n"
                "Generate a question batch first using the question generator."
            )
            return
        
        batch_names = [f.name for f in batch_files]
        
        # Use the latest generated batch if available
        latest_batch = st.session_state.get("latest_generated_batch")

        default_index = 0

        if latest_batch in batch_names:
            default_index = batch_names.index(latest_batch)

        # Batch selector
        selected_batch = st.selectbox(
            "Select Question Batch",
            options=batch_names,
            index=default_index,
        )
        
        if st.button("🔄 Refresh Batches", key="refresh_batches_button"):
            st.rerun()
    
    # Load the selected batch
    batch_data, error = load_batch(selected_batch)
    
    if error:
        st.error(f"❌ Error loading batch: {error}")
        return
    
    # Display batch summary
    st.subheader("Batch Summary")
    display_batch_summary(batch_data)
    
    # Filter by question type
    st.subheader("Filter Questions")
    
    available_types = get_available_question_types(batch_data)
    filter_options = ["All"] + available_types
    
    selected_filter = st.selectbox(
        "Question Type",
        options=filter_options,
        index=0,
        key="filter_selector"
    )
    
    # Filter questions based on selected type
    filtered_questions = filter_questions(batch_data, selected_filter)
    
    if not filtered_questions:
        st.warning("No questions match the selected filter.")
        return
    
    st.divider()
    
    # Display current question
    st.subheader("Question Review")
    
    # Ensure current index is valid for the filtered questions
    # Clamp to the valid range [0, len(filtered_questions) - 1]
    current_index = min(
        max(st.session_state.current_question_index, 0),
        len(filtered_questions) - 1
    )
    
    # If the index was out of bounds, update session state to the clamped value
    if current_index != st.session_state.current_question_index:
        st.session_state.current_question_index = current_index
    
    display_question(
        filtered_questions[current_index],
        current_index,
        len(filtered_questions),
    )
    
    # Navigation
    st.subheader("Navigation")
    display_navigation(current_index, len(filtered_questions))
    
    # Footer
    st.divider()
    st.markdown(
        "---\n"
        "**Note:** This is a read-only review interface. "
        "To regenerate questions or make changes, use the main question generator."
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Question Batch Reviewer",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    show_question_reviewer()
