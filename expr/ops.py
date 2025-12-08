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

'''
RL agent friendly mappings below
'''

RL_OPS = [("unary", "id"),("unary", "neg"),("unary", "exp"),("unary", "sin"),("unary", "cos"),("binary", "add"),("binary", "sub"),("binary", "mul"),("bianry", "div")]
RL_VOCAB_SIZE = len(RL_OPS)

UNARY_MASK = torch.tensor([op_type == "unary"  for (op_type, op_name) in RL_OPS], dtype=torch.bool)
BINARY_MASK = torch.tensor([op_type == "binary" for (op_type, op_name) in RL_OPS], dtype=torch.bool)