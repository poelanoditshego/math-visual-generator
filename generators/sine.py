import sympy as sp

from generators.trig_helpers import (
    TrigParameters,
    create_trig_graph,
    parse_trig_expression,
    phase_solutions,
)
from models.graph_settings import GraphSettings


SineParameters = TrigParameters
_phase_solutions = phase_solutions


def parse_sine_expression(
    equation: str,
) -> tuple[sp.Symbol, sp.Expr, TrigParameters]:
    return parse_trig_expression(equation, "Sine", sp.sin)


def create_sine_graph(equation: str, settings: GraphSettings) -> None:
    create_trig_graph(
        equation,
        settings,
        trig_name="Sine",
        trig_function=sp.sin,
        allow_pi_labels=False,
    )
