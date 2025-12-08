import torch

def id(x) : return x
def neg(x) : return -x
def add(a,b): return a+b
def sub(a,b): return a-b
def mul(a,b): return a*b
def div(a,b): return a/b


UNARY_OP = {
    "id" : id,
    "neg" : neg,
    "abs" : torch.abs,
    "exp" : torch.exp,
    "sin" : torch.sin,
    "cos" : torch.cos,
}

BINARY_OP = {
    "add" : add,
    "sub" : sub,
    "mul" : mul,
    "div" : div,
}

OP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "neg": "-",
    "id": "",
    "abs": "abs",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
}