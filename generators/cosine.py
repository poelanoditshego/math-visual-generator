import sympy as sp

from generators.trig_helpers import (
    TrigParameters,
    create_trig_graph,
    parse_trig_expression,
)
from models.graph_settings import GraphSettings


CosineParameters = TrigParameters


def parse_cosine_expression(
    equation: str,
) -> tuple[sp.Symbol, sp.Expr, TrigParameters]:
    return parse_trig_expression(equation, "Cosine", sp.cos)


def create_cosine_graph(equation: str, settings: GraphSettings) -> None:
    create_trig_graph(
        equation,
        settings,
        trig_name="Cosine",
        trig_function=sp.cos,
        allow_pi_labels=True,
    )
