import torch
import torch.nn as nn
import function as func

class nodeType:
    UNARY = 1
    ROOT = 2
    #Unary is implicit for leaves and nodes
    BINARY = 3
    LEAF = 4

UNARY = nodeType.UNARY
ROOT = nodeType.ROOT
BINARY = nodeType.BINARY
LEAF = nodeType.LEAF

class Node(nn.Module):
    def __init__(self, func = None, coeffs = None, weight = 1, bias = 0, arity = None):
        super().__init__()
        self.op = func
        self.children = []
        self.arity = arity
        #For leaves : coefficients
        self.coeffs = coeffs if coeffs is not None else torch.ones(1)
        self.weight = weight
        self.bias = bias

        self.opt_coeffs = nn.Parameter(self.coeffs.clone().detach().float())
        self.opt_weight = nn.Parameter(torch.tensor(float(self.weight)))
        self.opt_bias   = nn.Parameter(torch.tensor(float(self.bias)))

    @classmethod
    def newNode(cls, func, arity):
        return cls(func = func, coeffs = torch.ones(1), weight = 1, bias = 0, arity = arity)
    @classmethod
    def newLeaf(cls, func, coeffs, arity):
        return cls(func = func, coeffs = coeffs, weight = 1, bias = 0, arity = arity)
    def add_child(self, node : "Node"):
        self.children.append(node)
    

class ExpressionTree(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self._register_params(root)

    def _register_params(self, node):
        for i, child in enumerate(node.children):
            self.add_module(f"child_{id(child)}", child)
            self._register_params(child)
    def forward(self, x):
        return eval_tree(self.root, x)
    
def eval_node(node,x):
    if node.arity == LEAF:
        return torch.dot(node.op(x), node.coeffs)
    if not node.children:
        return(node.op(x))
    if node.arity == UNARY or node.arity == ROOT:
        return(node.weight*node.op(eval_node(node.children[0],x)) + node.bias)
    if node.arity == BINARY:
        values = [eval_node(c,x) for c in node.children]
        result = values[0]
        for v in values[1:]:
            result = node.op(result, v)
        return result

def eval_tree(tree, x):
    return eval_node(tree.root,x).item()


