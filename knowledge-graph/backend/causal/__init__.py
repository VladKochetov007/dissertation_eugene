"""
Causal inference package — DAG engine and Pearl's do-calculus.

Provides the algorithmic backbone for causal reasoning in the Republic of
AI Agents architecture, implementing Pearl's causal hierarchy from structural
graph operations to effect identification.
"""

from .dag import CausalDAGEngine
from .pearl import (
    backdoor_criterion,
    frontdoor_criterion,
    identify_effect,
    instrumental_variables,
)

__all__ = [
    "CausalDAGEngine",
    "backdoor_criterion",
    "frontdoor_criterion",
    "identify_effect",
    "instrumental_variables",
]
