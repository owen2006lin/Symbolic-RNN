import torch


UNARY_OP = {
    "id" : lambda x : x,
    "neg" : lambda x : -x,
    "abs" : torch.abs,
    "exp" : torch.exp,
    "sin" : torch.sin,
    "cos" : torch.cos,
}

BINARY_OP = {
    "add" : lambda a, b : a + b,
    "sub" : lambda a, b : a - b,
    "mul" : lambda a, b : a * b,
    "div" : lambda a, b : a / b,
}