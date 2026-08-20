import unittest

from generators.graph_helpers import parse_arithmetic_expression
from generators.linear import create_linear_graph
from generators.mixed import parse_mixed_expression
from generators.quadratic import create_quadratic_graph
from models.graph_settings import GraphSettings


class EquationParsingTests(unittest.TestCase):
    def setUp(self):
        self.settings = GraphSettings(output_name="parsing-test.png")

    def test_shared_parser_accepts_expression_and_y_form(self):
        _, expression = parse_arithmetic_expression(
            "y = 2*x + 1", "Linear", "2*x + 1"
        )
        self.assertEqual(str(expression), "2*x + 1")

    def test_linear_accepts_y_form(self):
        create_linear_graph("y = 2*x + 1", self.settings)

    def test_quadratic_accepts_y_form(self):
        create_quadratic_graph("y = x**2 - 4", self.settings)

    def test_invalid_variables_and_functions_are_rejected(self):
        for equation in ("2*z + 1", "x + y"):
            with self.subTest(equation=equation), self.assertRaisesRegex(
                ValueError, "Only x is allowed"
            ):
                parse_arithmetic_expression(equation, "Linear", "2*x + 1")

        with self.assertRaisesRegex(ValueError, "Unsupported function"):
            parse_arithmetic_expression("sin(x)", "Linear", "2*x + 1")

    def test_invalid_polynomial_types_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "linear function"):
            create_linear_graph("x**2", self.settings)
        with self.assertRaisesRegex(ValueError, "quadratic function"):
            create_quadratic_graph("x", self.settings)

    def test_mixed_uses_shared_y_form_parser(self):
        self.assertEqual(parse_mixed_expression("y = 2*x + 1", 1)[2], "Linear")
        self.assertEqual(
            parse_mixed_expression("y = 2/(x - 1) + 3", 2)[2], "Hyperbola"
        )

    def test_malformed_expression_is_controlled(self):
        with self.assertRaisesRegex(ValueError, "could not be understood"):
            parse_arithmetic_expression("2**(", "Linear", "2*x + 1")


if __name__ == "__main__":
    unittest.main()