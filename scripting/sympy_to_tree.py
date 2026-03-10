"""
sympy_to_tree.py

Converts a SymPy expression into the project's ExpressionTree / Node format
defined in expr/tree.py.

Canonicalization rules (from treebuilding.ipynb):
  - Add/Mul associativity is already flattened by SymPy; mapped to n-ary BINARY nodes
  - Subtraction (Add with negative terms) → BINARY(add) with UNARY(neg) children
  - Division (Mul with Pow(x,-1) terms) → BINARY(div)
  - Pow(x, 1/2) → UNARY(sqrt)
  - Pow(x, n) for positive integer n ≤ 4 → n-ary BINARY(mul)
  - Pow(x, n) otherwise → BINARY(pow)
  - Known functions → UNARY with corresponding op from UNARY_OP
  - Symbols → symbolic LEAF (Node.symbolicLeaf)
  - Numbers → symbolic LEAF whose name is the number string
  - Derivatives → already replaced with Symbol by latex_cleaner

Raises UnsupportedExprError for any SymPy type not covered by the above.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the scripting/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expr.tree import Node, ExpressionTree, UNARY, BINARY, LEAF, ROOT
from expr.ops import UNARY_OP, BINARY_OP


class UnsupportedExprError(ValueError):
    """Raised when a SymPy expression contains unsupported structure."""
    pass


# ── SymPy function class → UNARY_OP key ──────────────────────────────────────
# Built lazily at first use to avoid importing SymPy at module load time.
_SYMPY_UNARY_MAP: dict | None = None


def _get_unary_map() -> dict:
    global _SYMPY_UNARY_MAP
    if _SYMPY_UNARY_MAP is None:
        from sympy import sin, cos, tan, exp, log, Abs, asin, acos, atan, sinh, cosh, tanh
        _SYMPY_UNARY_MAP = {
            sin:  'sin',
            cos:  'cos',
            tan:  'tan',
            exp:  'exp',
            log:  'log',
            Abs:  'abs',
            asin: 'arcsin',
            acos: 'arccos',
            atan: 'arctan',
            sinh: 'sinh',
            cosh: 'cosh',
            tanh: 'tanh',
        }
    return _SYMPY_UNARY_MAP


# ── Mul handler ───────────────────────────────────────────────────────────────

def _mul_to_node(args) -> Node:
    """
    Convert the args of a SymPy Mul to an ExprTree node.

    Handles:
      Mul(-1, x)           → UNARY(neg)(x)
      Mul(x, Pow(y, -1))   → BINARY(div)(x, y)
      Mul(-1, x, Pow(y,-1))→ UNARY(neg)(BINARY(div)(x, y))
      Mul(a, b, c)         → BINARY(mul)(a, b, c)   [n-ary]
      Mul(-2, x)           → UNARY(neg)(BINARY(mul)(leaf("2"), x))
    """
    from sympy import Number, Pow, S

    is_neg = False
    numerator: list = []
    denominator: list = []

    for arg in args:
        if isinstance(arg, Number):
            val = float(arg)
            if val < 0:
                is_neg = not is_neg       # track sign separately
                if abs(val) != 1.0:
                    numerator.append(-arg)  # keep the absolute value as a factor
            else:
                if val != 1.0:            # skip trivial ×1
                    numerator.append(arg)
        elif isinstance(arg, Pow) and arg.args[1] == S.NegativeOne:
            denominator.append(arg.args[0])
        else:
            numerator.append(arg)

    # Build numerator subtree
    if not numerator:
        num_node = Node.symbolicLeaf('1')
    elif len(numerator) == 1:
        num_node = sympy_to_node(numerator[0])
    else:
        num_node = Node.newNode(BINARY_OP['mul'], BINARY)
        for a in numerator:
            num_node.add_child(sympy_to_node(a))

    # Attach denominator if present
    if not denominator:
        result = num_node
    elif len(denominator) == 1:
        result = Node.newNode(BINARY_OP['div'], BINARY)
        result.add_child(num_node)
        result.add_child(sympy_to_node(denominator[0]))
    else:
        den_node = Node.newNode(BINARY_OP['mul'], BINARY)
        for a in denominator:
            den_node.add_child(sympy_to_node(a))
        result = Node.newNode(BINARY_OP['div'], BINARY)
        result.add_child(num_node)
        result.add_child(den_node)

    # Wrap in neg if sign was negative
    if is_neg:
        neg_node = Node.newNode(UNARY_OP['neg'], UNARY)
        neg_node.add_child(result)
        return neg_node

    return result


# ── Pow handler ───────────────────────────────────────────────────────────────

def _pow_to_node(base, exp) -> Node:
    """
    Convert Pow(base, exp) to an ExprTree node.

      Pow(x, 1/2)          → UNARY(sqrt)(x)
      Pow(x, -1)           → BINARY(div)(leaf("1"), x)
      Pow(x, n)  n∈[2,4]  → n-ary BINARY(mul)(x, x, ..., x)
      Pow(x, -n) n∈[2,4]  → BINARY(div)(leaf("1"), BINARY(mul)(x,...,x))
      Pow(x, n)  otherwise → BINARY(pow)(x, n)
    """
    from sympy import Integer, S

    # sqrt
    if exp == S.Half:
        node = Node.newNode(UNARY_OP['sqrt'], UNARY)
        node.add_child(sympy_to_node(base))
        return node

    # x^(-1) = 1/x  (standalone; normally absorbed by _mul_to_node)
    if exp == S.NegativeOne:
        div_node = Node.newNode(BINARY_OP['div'], BINARY)
        div_node.add_child(Node.symbolicLeaf('1'))
        div_node.add_child(sympy_to_node(base))
        return div_node

    # Integer powers
    if isinstance(exp, Integer):
        n = int(exp)
        if 2 <= n <= 4:
            mul_node = Node.newNode(BINARY_OP['mul'], BINARY)
            for _ in range(n):
                mul_node.add_child(sympy_to_node(base))
            return mul_node
        if -4 <= n <= -2:
            expanded = Node.newNode(BINARY_OP['mul'], BINARY)
            for _ in range(-n):
                expanded.add_child(sympy_to_node(base))
            div_node = Node.newNode(BINARY_OP['div'], BINARY)
            div_node.add_child(Node.symbolicLeaf('1'))
            div_node.add_child(expanded)
            return div_node
        if n == 1:
            return sympy_to_node(base)

    # General case: BINARY(pow)
    pow_node = Node.newNode(BINARY_OP['pow'], BINARY)
    pow_node.add_child(sympy_to_node(base))
    pow_node.add_child(sympy_to_node(exp))
    return pow_node


# ── Main recursive converter ──────────────────────────────────────────────────

def sympy_to_node(expr) -> Node:
    """
    Recursively convert a SymPy expression to a Node tree.
    Raises UnsupportedExprError for unsupported expression types.
    """
    from sympy import Add, Mul, Pow, Symbol, Number, S

    # NaN / zoo / oo — not representable
    if expr in (S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
        raise UnsupportedExprError(f"Non-finite value: {expr}")

    # Symbolic constants → named leaf
    if expr is S.Pi:
        return Node.symbolicLeaf('pi')
    if expr is S.Exp1:
        return Node.symbolicLeaf('e')
    if expr is S.ImaginaryUnit:
        return Node.symbolicLeaf('i')

    # Symbol or Number → leaf
    if isinstance(expr, (Symbol, Number)):
        return Node.symbolicLeaf(str(expr))

    # Add → n-ary BINARY(add)  (SymPy already flattens nested Adds)
    if isinstance(expr, Add):
        node = Node.newNode(BINARY_OP['add'], BINARY)
        for arg in expr.args:
            node.add_child(sympy_to_node(arg))
        return node

    # Mul → handled separately (neg / div extraction)
    if isinstance(expr, Mul):
        return _mul_to_node(expr.args)

    # Pow
    if isinstance(expr, Pow):
        return _pow_to_node(expr.args[0], expr.args[1])

    # Known unary functions
    for sympy_type, op_name in _get_unary_map().items():
        if isinstance(expr, sympy_type):
            node = Node.newNode(UNARY_OP[op_name], UNARY)
            node.add_child(sympy_to_node(expr.args[0]))
            return node

    raise UnsupportedExprError(
        f"Unsupported SymPy type '{type(expr).__name__}': {expr}"
    )


def sympy_to_tree(expr) -> ExpressionTree:
    """
    Convert a SymPy expression to an ExpressionTree.

    Wraps the converted node in a ROOT(id) node, consistent with the
    project's tree construction convention (see expr/tree.py).

    Raises UnsupportedExprError if the expression contains unsupported ops.
    """
    root = Node.newNode(UNARY_OP['id'], ROOT)
    root.add_child(sympy_to_node(expr))
    return ExpressionTree(root)


# ── Quick smoke-test when run as a script ─────────────────────────────────────

if __name__ == '__main__':
    from sympy import sympify, symbols, sin, cos, exp, log, sqrt, Abs

    x, y, z = symbols('x y z')

    test_cases = [
        (sin(x) + cos(y),                  'sin(x) + cos(y)'),
        (x**2 + 2*x + 1,                   'x^2 + 2x + 1  (poly)'),
        ((x**2 + 1) / (x - 1),             'frac'),
        (-x / y,                            '-x/y  (neg + div)'),
        (exp(log(x)) + sqrt(y),             'exp(log(x)) + sqrt(y)'),
        (x**5,                              'x^5  (general pow)'),
        (Abs(x - y),                        'abs(x-y)'),
        (x * y * z,                         'x*y*z  (n-ary mul)'),
    ]

    for expr, label in test_cases:
        try:
            tree = sympy_to_tree(expr)
            print(f'  OK  [{label}]')
            print(f'       str : {tree.to_str()}')
            tree.ordered_print()
        except UnsupportedExprError as e:
            print(f'  ERR [{label}]: {e}')
