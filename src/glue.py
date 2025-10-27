"""
M6 GLUE — Single composition of per-class actions.

Exports glue_once from M5's fy.py, which already implements the correct GLUE formula:
Ŷ' = X' ⊕ ⊕_k (A_k(X') ⊙ M_k(X'))

Each action applies to the ORIGINAL X', not the accumulated result.
Masks are disjoint by construction, so order is irrelevant (but we use sorted order for determinism).
"""

from src.fy import glue_once

__all__ = ['glue_once']
