"""Tests for Linear question display policy and information sufficiency."""

import pytest
from models.question_models import QuestionBlueprint, LinearQuestionData
from questions.linear import build_linear_display_settings, build_memo


class TestLinearDisplayPolicy:
    """Tests for Linear question display settings policy."""
    
    def test_x_intercept_hides_answer_and_graph_text(self):
        """Test x_intercept: hide answer marker and learner graph text."""
        display = build_linear_display_settings("x_intercept")
        
        # Hide the direct answer (x-intercept marker)
        assert display.show_x_intercepts is False, "x-intercept should be hidden (it's the answer)"
        
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False
        
        # Show y-intercept for reference
        assert display.show_y_intercepts is True, "y-intercept should be shown for reference"
    
    def test_y_intercept_hides_answer_and_graph_text(self):
        """Test y_intercept: hide answer marker and learner graph text."""
        display = build_linear_display_settings("y_intercept")
        
        # Hide the direct answer (y-intercept marker)
        assert display.show_y_intercepts is False, "y-intercept should be hidden (it's the answer)"
        
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False
        
        # Show x-intercept for reference
        assert display.show_x_intercepts is True, "x-intercept should be shown for reference"
    
    def test_gradient_hides_answer_and_graph_annotations(self):
        """Test gradient: hide all graph presentation of the gradient."""
        display = build_linear_display_settings("gradient")
        
        # Hide gradient annotations and triangle (they show the answer)
        assert display.show_gradient is False, "gradient should be hidden (it's the answer)"
        assert display.show_gradient_triangle is False, "gradient triangle should be hidden"
        
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False
        
        # Show intercepts for reference
        assert display.show_x_intercepts is True, "x-intercept should be shown for reference"
        assert display.show_y_intercepts is True, "y-intercept should be shown for reference"
    
    def test_future_determine_equation_type_has_hidden_equation(self):
        """Test that future equation questions hide the equation everywhere."""
        display = build_linear_display_settings("determine_equation")
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False

    def test_unsupported_question_type_raises(self):
        """Test that unsupported question types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported linear question type"):
            build_linear_display_settings("quadratic")


class TestLinearMemoExplainsGraphMethod:
    """Tests for Linear memo generation using graph-based explanations."""
    
    def test_x_intercept_memo_uses_graph_equation(self):
        """Test x_intercept memo explains using the visible equation."""
        data = LinearQuestionData(
            equation="2*x - 4",
            gradient=2,
            y_intercept=-4,
            x_intercept=2,
        )
        
        answer, memo = build_memo("x_intercept", data)
        
        assert answer == "(2, 0)"
        # Memo should reference the equation shown on the graph
        assert "equation" in memo.lower()
        assert "2*x - 4" in memo or "2x - 4" in memo
        # Memo should explain the graph-based method
        assert "x-axis" in memo.lower() or "y = 0" in memo
        # Should NOT refer to hidden information
        assert "coefficient" not in memo.lower()
    
    def test_y_intercept_memo_uses_graph_equation(self):
        """Test y_intercept memo explains using the visible equation."""
        data = LinearQuestionData(
            equation="2*x + 3",
            gradient=2,
            y_intercept=3,
            x_intercept=-1.5,
        )
        
        answer, memo = build_memo("y_intercept", data)
        
        assert answer == "(0, 3)"
        # Memo should reference the equation shown on the graph
        assert "equation" in memo.lower()
        assert "2*x + 3" in memo or "2x + 3" in memo
        # Memo should explain the graph-based method
        assert "y-axis" in memo.lower() or "x = 0" in memo
        # Should NOT refer to hidden information
        assert "coefficient" not in memo.lower()
    
    def test_gradient_memo_uses_graph_equation(self):
        """Test gradient memo explains using the visible equation."""
        data = LinearQuestionData(
            equation="-3*x + 5",
            gradient=-3,
            y_intercept=5,
            x_intercept=5/3,
        )
        
        answer, memo = build_memo("gradient", data)
        
        assert answer == "-3"
        # Memo should reference the equation shown on the graph
        assert "equation" in memo.lower()
        assert "-3*x + 5" in memo or "-3x + 5" in memo
        # Memo should explain the coefficient method
        assert "coefficient" in memo.lower()
        assert "mx + c" in memo or "y = mx + c" in memo
    
    def test_memo_formatted_correctly(self):
        """Test that memos are well-formatted and readable."""
        data = LinearQuestionData(
            equation="x + 1",
            gradient=1,
            y_intercept=1,
            x_intercept=-1,
        )
        
        for question_type in ["x_intercept", "y_intercept", "gradient"]:
            answer, memo = build_memo(question_type, data)
            
            # Answer should not be empty
            assert answer, f"Answer should not be empty for {question_type}"
            
            # Memo should not be empty
            assert memo, f"Memo should not be empty for {question_type}"
            
            # Memo should be well-formatted (contain newlines, multiple sentences)
            assert "\n" in memo, f"Memo should be multi-line for {question_type}"


class TestLinearQuestionInformationSufficiency:
    """Tests asserting sufficient information for each question type."""
    
    def test_x_intercept_question_has_sufficient_info(self):
        """Test that x_intercept questions provide enough info to solve."""
        display = build_linear_display_settings("x_intercept")
        
        assert display.show_equation is False
        
        # For a learner to find x-intercept:
        # - They need the equation (they have it)
        # - They DON'T need to see the x-intercept marked (they calculate it)
        assert display.show_x_intercepts is False
    
    def test_y_intercept_question_has_sufficient_info(self):
        """Test that y_intercept questions provide enough info to solve."""
        display = build_linear_display_settings("y_intercept")
        
        assert display.show_equation is False
        
        # For a learner to find y-intercept:
        # - They need the equation (they have it)
        # - They DON'T need to see the y-intercept marked (they calculate it)
        assert display.show_y_intercepts is False
    
    def test_gradient_question_has_sufficient_info(self):
        """Test that gradient questions provide enough info to solve."""
        display = build_linear_display_settings("gradient")
        
        assert display.show_equation is False
        
        # For a learner to find gradient:
        # - They need the equation (they have it)
        # - They DON'T need the gradient triangle or annotation (they read it from equation)
        assert display.show_gradient is False
        assert display.show_gradient_triangle is False
    
    def test_all_types_hide_graph_text_for_learner_presentation(self):
        """Test that learner graphs have no equation, title, or legend."""
        for question_type in ["x_intercept", "y_intercept", "gradient"]:
            display = build_linear_display_settings(question_type)
            assert display.show_equation is False
            assert display.show_title is False
            assert display.show_legend is False

    def test_determine_equation_hides_equation_everywhere(self):
        display = build_linear_display_settings("determine_equation")
        assert display.show_equation is False
        assert display.show_title is False
        assert display.show_legend is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
