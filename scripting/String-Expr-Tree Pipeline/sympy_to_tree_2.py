"""
sympy_to_tree.py

Converts a SymPy expression into the project's ExpressionTree / Node format
defined in expr/tree.py.

Canonicalization rules:
  - Add/Mul associativity is already flattened by SymPy; mapped to n-ary nodes
  - Subtraction (Add with negative terms) -> BINARY(add) with UNARY(neg) children
  - Division (Mul with negative integer powers) -> BINARY(div)
  - Pow(x, 1/2) -> UNARY(sqrt)(x)
  - Pow(x, n) for integer n in {2,3,4} -> UNARY(pow2/pow3/pow4)(x)
  - Pow(x, -n) for integer n in {2,3,4} -> BINARY(div)(1, UNARY(powN)(x))
  - Pow(x, n) otherwise -> BINARY(pow)(x, n)
  - Known functions -> UNARY with corresponding op from UNARY_OP
  - Symbols -> symbolic LEAF (Node.symbolicLeaf)
  - Numbers -> symbolic LEAF whose name is the number string
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from anywhere: add project root so that expr.tree / expr.ops
# are importable.  File lives at <root>/scripting/String-Expr-Tree Pipeline/,
# so three .parent calls reach <root>.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from expr.tree import Node, ExpressionTree, UNARY, BINARY, ROOT
from expr.ops import UNARY_OP, BINARY_OP


class UnsupportedExprError(ValueError):
    """Raised when a SymPy expression contains unsupported structure."""
    pass


# Built lazily at first use to avoid importing SymPy at module load time.
_SYMPY_UNARY_MAP: dict | None = None


def _get_unary_map() -> dict:
    global _SYMPY_UNARY_MAP
    if _SYMPY_UNARY_MAP is None:
        from sympy import sin, cos, tan, exp, log, Abs, asin, acos, atan, sinh, cosh, tanh

        _SYMPY_UNARY_MAP = {
            sin:  "sin",
            cos:  "cos",
            tan:  "tan",
            exp:  "exp",
            log:  "log",
            Abs:  "abs",
            asin: "arcsin",
            acos: "arccos",
            atan: "arctan",
            sinh: "sinh",
            cosh: "cosh",
            tanh: "tanh",
        }
    return _SYMPY_UNARY_MAP


def _mul_from_nodes(nodes: list[Node]) -> Node:
    if not nodes:
        return Node.symbolicLeaf("1")
    if len(nodes) == 1:
        return nodes[0]

    mul_node = Node.newNode(BINARY_OP["mul"], BINARY)
    for node in nodes:
        mul_node.add_child(node)
    return mul_node


def _div_node(num: Node, den: Node) -> Node:
    node = Node.newNode(BINARY_OP["div"], BINARY)
    node.add_child(num)
    node.add_child(den)
    return node


def _positive_power_node(base, n: int) -> Node:
    """
    Build a node for base^n where n is a positive integer.
    Uses unary pow2/pow3/pow4 for n in {2,3,4}, otherwise binary pow.
    """
    if n == 1:
        return sympy_to_node(base)

    if n in (2, 3, 4):
        node = Node.newNode(UNARY_OP[f"pow{n}"], UNARY)
        node.add_child(sympy_to_node(base))
        return node

    pow_node = Node.newNode(BINARY_OP["pow"], BINARY)
    pow_node.add_child(sympy_to_node(base))
    pow_node.add_child(Node.symbolicLeaf(str(n)))
    return pow_node


def _pow_to_node(base, exp) -> Node:
    """
    Convert Pow(base, exp) to an ExprTree node.

      Pow(x, 1/2)            -> UNARY(sqrt)(x)
      Pow(x, -1)             -> BINARY(div)(1, x)
      Pow(x, n), n=2,3,4     -> UNARY(pow2/pow3/pow4)(x)
      Pow(x, -n), n=2,3,4    -> BINARY(div)(1, UNARY(powN)(x))
      Pow(x, n) otherwise    -> BINARY(pow)(x, n)
    """
    from sympy import Integer, S

    # sqrt(x)
    if exp == S.Half:
        node = Node.newNode(UNARY_OP["sqrt"], UNARY)
        node.add_child(sympy_to_node(base))
        return node

    # integer powers
    if isinstance(exp, Integer):
        n = int(exp)

        if n == 1:
            return sympy_to_node(base)

        if n == -1:
            return _div_node(Node.symbolicLeaf("1"), sympy_to_node(base))

        if n > 1:
            return _positive_power_node(base, n)

        if n < -1:
            return _div_node(Node.symbolicLeaf("1"), _positive_power_node(base, -n))

    # general case
    pow_node = Node.newNode(BINARY_OP["pow"], BINARY)
    pow_node.add_child(sympy_to_node(base))
    pow_node.add_child(sympy_to_node(exp))
    return pow_node


def _mul_to_node(args) -> Node:
    """
    Convert the args of a SymPy Mul to an ExprTree node.

    Handles:
      Mul(-1, x)             -> UNARY(neg)(x)
      Mul(x, y**-1)          -> BINARY(div)(x, y)
      Mul(x, y**-2)          -> BINARY(div)(x, pow2(y))
      Mul(-1, x, y**-3)      -> UNARY(neg)(BINARY(div)(x, pow3(y)))
      Mul(a, b, c)           -> BINARY(mul)(a, b, c)
      Mul(-2, x)             -> UNARY(neg)(BINARY(mul)(2, x))
    """
    from sympy import Number, Pow, Integer

    is_neg = False
    numerator_nodes: list[Node] = []
    denominator_nodes: list[Node] = []

    for arg in args:
        # numeric factors: pull sign out separately
        if isinstance(arg, Number):
            if arg.is_negative:
                is_neg = not is_neg
                abs_arg = -arg
                if abs_arg != 1:
                    numerator_nodes.append(sympy_to_node(abs_arg))
            else:
                if arg != 1:
                    numerator_nodes.append(sympy_to_node(arg))
            continue

        # negative integer powers go to denominator
        if isinstance(arg, Pow):
            base, exp = arg.args
            if isinstance(exp, Integer) and int(exp) < 0:
                n = -int(exp)
                if n == 1:
                    denominator_nodes.append(sympy_to_node(base))
                else:
                    denominator_nodes.append(_positive_power_node(base, n))
                continue

        numerator_nodes.append(sympy_to_node(arg))

    num_node = _mul_from_nodes(numerator_nodes)

    if denominator_nodes:
        den_node = _mul_from_nodes(denominator_nodes)
        result = _div_node(num_node, den_node)
    else:
        result = num_node

    if is_neg:
        neg_node = Node.newNode(UNARY_OP["neg"], UNARY)
        neg_node.add_child(result)
        return neg_node

    return result


def sympy_to_node(expr) -> Node:
    """
    Recursively convert a SymPy expression to a Node tree.
    Raises UnsupportedExprError for unsupported expression types.
    """
    from sympy import Add, Mul, Pow, Symbol, Number, S

    # non-finite values not representable
    if expr in (S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
        raise UnsupportedExprError(f"Non-finite value: {expr}")

    # symbolic constants
    if expr is S.Pi:
        return Node.symbolicLeaf("pi")
    if expr is S.Exp1:
        return Node.symbolicLeaf("e")
    if expr is S.ImaginaryUnit:
        return Node.symbolicLeaf("i")

    # symbols / numbers
    if isinstance(expr, (Symbol, Number)):
        return Node.symbolicLeaf(str(expr))

    # add
    if isinstance(expr, Add):
        node = Node.newNode(BINARY_OP["add"], BINARY)
        for arg in expr.args:
            node.add_child(sympy_to_node(arg))
        return node

    # mul
    if isinstance(expr, Mul):
        return _mul_to_node(expr.args)

    # pow
    if isinstance(expr, Pow):
        return _pow_to_node(expr.args[0], expr.args[1])

    # known unary functions
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
    project's tree construction convention.
    """
    root = Node.newNode(UNARY_OP["id"], ROOT)
    root.add_child(sympy_to_node(expr))
    return ExpressionTree(root)


if __name__ == "__main__":
    from sympy import symbols, sin, cos, exp, log, sqrt, Abs

    x, y, z = symbols("x y z")

    test_cases = [
        (sin(x) + cos(y),          "sin(x) + cos(y)"),
        (x**2 + 2*x + 1,           "x^2 + 2x + 1 (poly)"),
        ((x**2 + 1) / (x - 1),     "frac"),
        (-x / y,                   "-x/y (neg + div)"),
        (exp(log(x)) + sqrt(y),    "exp(log(x)) + sqrt(y)"),
        (x**5,                     "x^5 (general pow)"),
        (Abs(x - y),               "abs(x-y)"),
        (x * y * z,                "x*y*z (n-ary mul)"),
        (x * y**-2,                "x / y^2"),
        (x**-3,                    "1 / x^3"),
    ]

    for expr, label in test_cases:
        try:
            tree = sympy_to_tree(expr)
            print(f"  OK  [{label}]")
            print(f"       str : {tree.to_str()}")
            tree.ordered_print()
        except UnsupportedExprError as e:
            print(f"  ERR [{label}]: {e}")