"""
scalar_filter.py

Strips scalar/constant terms from SymPy expressions.

Scalar definition
-----------------
An expression is "scalar" (constant) if ``expr.free_symbols == set()``.
This includes:
  - Raw numbers:          2, 3/7, 0.5
  - Named SymPy constants: pi, E (Euler's number), I (imaginary unit)
  - Combinations thereof: pi/2, 2*pi, E**2, sqrt(2)

Raw Symbol objects are NEVER scalar — they always carry free symbols by
definition, even if they represent a constant in the original paper.

Filtering rules (top-level only — not inside function arguments)
----------------------------------------------------------------
1. **Add**: drop additive terms where ``term.free_symbols == set()``.
2. **Mul**: drop multiplicative factors where ``factor.free_symbols == set()``.
   Applied in sequence: strip additive constants first, then strip
   multiplicative scalars from the result.

Examples
--------
  x + 3            →  x
  sin(x) + pi/2    →  sin(x)
  2 * x            →  x
  -sin(x)          →  sin(x)
  (3/7)*sin(x) + 4 →  sin(x)
  x * y            →  x * y   (no scalars at top level)
  sin(x + 3)       →  sin(x + 3)   (3 is inside a function arg, not top-level)

Sign policy
-----------
Negation is represented in SymPy as ``Mul(-1, expr)``.  Because -1 is scalar
(no free symbols), it is dropped by the Mul rule.  Therefore:
  -x       →  x
  -sin(x)  →  sin(x)
This is intentional: the project assumes linear transformations (a·f + b) are
applied externally, so sign and coefficient are considered irrelevant.

Fallback
--------
If filtering would eliminate the entire expression (all terms/factors are
scalar, or the input already has no free symbols), ``filter_scalars`` returns
the original expression unchanged.  A call like ``filter_scalars(S(3))``
returns ``S(3)`` rather than failing.

Public API
----------
filter_scalars(expr) -> expr
    Main entry point.  Always returns a SymPy expression.
"""

from __future__ import annotations

from typing import Optional

from sympy import Add, Mul, Expr


# ── Internal helper ───────────────────────────────────────────────────────────

def _filter_impl(expr) -> Optional[object]:
    """
    Core filter.  Returns the filtered SymPy expression, or None if the entire
    expression is scalar (so the caller should use the original as the fallback).
    """
    # Already entirely scalar → signal removal to caller
    if not expr.free_symbols:
        return None

    # ── Rule 1: Remove constant additive terms ────────────────────────────────
    if isinstance(expr, Add):
        non_const = [a for a in expr.args if a.free_symbols]
        if not non_const:
            return None
        if len(non_const) == 1:
            expr = non_const[0]        # continue to Mul check below
        elif len(non_const) < len(expr.args):
            expr = Add(*non_const)     # rebuild without constant terms
        # else: nothing was constant in the Add, fall through

    # ── Rule 2: Remove constant multiplicative factors (incl. sign) ───────────
    if isinstance(expr, Mul):
        non_const = [f for f in expr.args if f.free_symbols]
        if not non_const:
            return None
        if len(non_const) == 1:
            return non_const[0]
        if len(non_const) < len(expr.args):
            return Mul(*non_const)

    return expr


# ── Public API ────────────────────────────────────────────────────────────────

def filter_scalars(expr) -> object:
    """
    Remove scalar/constant additive and multiplicative terms from *expr*.

    Returns the filtered SymPy expression.  If filtering would reduce the
    expression to nothing (every term/factor is scalar), returns *expr*
    unchanged (documented fallback).

    See module docstring for the full scalar definition and sign policy.
    """
    result = _filter_impl(expr)
    return expr if result is None else result
