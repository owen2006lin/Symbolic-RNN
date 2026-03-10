import torch

# ----------
# Basic ops
# ----------

def id(x): return x
def neg(x): return -x
def abs_op(x): return torch.abs(x)
def exp_op(x): return torch.exp(x)
def log_op(x): return torch.log(x)
def sqrt_op(x): return torch.sqrt(x)
def sin_op(x): return torch.sin(x)
def cos_op(x): return torch.cos(x)
def tan_op(x): return torch.tan(x)
def arcsin_op(x): return torch.arcsin(x)
def arccos_op(x): return torch.arccos(x)
def arctan_op(x): return torch.arctan(x)
def sinh_op(x): return torch.sinh(x)
def cosh_op(x): return torch.cosh(x)
def tanh_op(x): return torch.tanh(x)
def pow2(x): return x * x
def pow3(x): return x * x * x
def pow4(x): pow2(x) * pow2(x)

# ----------
# Binary ops
# ----------

def add(a, b): return a + b

def mul(a, b): return a * b

def div(a, b): return a / b

def pow_op(a, b): return a ** b


UNARY_OP = {
    "id"     : id,
    "neg"    : neg,
    "abs"    : abs_op,
    "exp"    : exp_op,
    "log"    : log_op,
    "sqrt"   : sqrt_op,
    "sin"    : sin_op,
    "cos"    : cos_op,
    "tan"    : tan_op,
    "arcsin" : arcsin_op,
    "arccos" : arccos_op,
    "arctan" : arctan_op,
    "sinh"   : sinh_op,
    "cosh"   : cosh_op,
    "tanh"   : tanh_op,
    "pow2"   : pow2,
    "pow3"   : pow3,
    "pow4"   : pow4,
}

BINARY_OP = {
    "add" : add,
    "mul" : mul,
    "div" : div,
    "pow" : pow_op,
}

OP_SYMBOLS = {
    "add"    : "+",
    "mul"    : "*",
    "div"    : "/",
    "pow"    : "**",
    "neg"    : "-",
    "id"     : "",
    "abs"    : "abs",
    "exp"    : "exp",
    "log"    : "log",
    "sqrt"   : "sqrt",
    "sin"    : "sin",
    "cos"    : "cos",
    "tan"    : "tan",
    "arcsin" : "arcsin",
    "arccos" : "arccos",
    "arctan" : "arctan",
    "sinh"   : "sinh",
    "cosh"   : "cosh",
    "tanh"   : "tanh",
    "pow2"   : "^2",
    "pow3"   : "^3",
    "pow4"   : "^4",
}

"""
RL agent friendly mappings below
"""

RL_OPS = [
    ("unary",  "id"),
    ("unary",  "neg"),
    ("unary",  "exp"),
    ("unary",  "sin"),
    ("unary",  "cos"),
    ("unary",  "pow2"),
    ("unary",  "pow3"),
    ("unary",  "pow4"),
    ("binary", "add"),
    ("binary", "sub"),
    ("binary", "mul"),
    ("binary", "div"),
]

RL_VOCAB_SIZE = len(RL_OPS)

UNARY_MASK = torch.tensor(
    [op_type == "unary" for (op_type, op_name) in RL_OPS],
    dtype=torch.bool
)

BINARY_MASK = torch.tensor(
    [op_type == "binary" for (op_type, op_name) in RL_OPS],
    dtype=torch.bool
)