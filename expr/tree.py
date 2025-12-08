import torch
import torch.nn as nn
from collections import deque
from .ops import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.coeffs = nn.Parameter(coeffs if coeffs is not None else torch.ones(1))
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.bias   = nn.Parameter(torch.tensor(float(bias)))
    
    #for creating unbiased root or weighted nonleaf nodes with no bias
    @classmethod
    def newNode(cls, func, arity):
        return cls(func = func, coeffs = torch.ones(1), weight = 1, bias = 0, arity = arity)
    
    #for creating nonleaf nodes with a nonzero bias and weight
    @classmethod
    def weightedNode(cls, func, weight, bias, arity):
        return cls(func = func, coeffs = torch.ones(1), weight = weight, bias = bias, arity = arity)
    
    #creating leaf node with coefficients
    @classmethod
    def newLeaf(cls, func, coeffs, arity):
        return cls(func = func, coeffs = coeffs, weight = 1, bias = 0, arity = arity)
    def add_child(self, node : "Node"):
        self.children.append(node)
    

def node_to_str(node):
    # Leaf node
    if node.arity == LEAF:
        name = None
        for n, f in {**UNARY_OP, **BINARY_OP}.items():
            if f == node.op:
                name = n
                break
        if name is None:
            name = getattr(node.op, "__name__", "x")
        if name == "id":
            return "x"
        else:
            return f"{name}(x)"

    # Unary or root node (non-leaf)
    elif node.arity in [UNARY, ROOT]:
        child_str = node_to_str(node.children[0])
        w = node.weight.item() if hasattr(node, "weight") else 1
        b = node.bias.item() if hasattr(node, "bias") else 0
        expr = child_str
        if w != 1:
            expr = f"{w}*({expr})"
        if b != 0:
            expr = f"{expr} + {b}"
        return expr

    # Binary node with any number of children
    elif node.arity == BINARY:
        symbol = None
        for n, f in BINARY_OP.items():
            if f == node.op:
                symbol = OP_SYMBOLS.get(n, n)
                break
        if symbol is None:
            symbol = "?"
        children_strs = [node_to_str(c) for c in node.children]
        return "(" + f" {symbol} ".join(children_strs) + ")"


def ordered_print(node, prefix="", is_last=True):
    # Determine which branch symbol to use
    connector = "└── " if is_last else "├── "
    print(prefix + connector + (node.op.__name__ if hasattr(node.op, "__name__") else str(node.op)))

    # Prepare the prefix for child nodes
    if node.children:
        # Add '    ' if this is the last child, '│   ' if not
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            ordered_print(child, prefix=new_prefix, is_last=(i == len(node.children)-1))

            
def to_sequence(node):
    seq = []
    queue = deque([node])
    while queue:
        n = queue.popleft()
        op_name = n.op.__name__
        seq.append(op_name)
        for child in n.children:
            queue.append(child)
    return seq

def move_node_to_device(node, device):
    # Move parameters
    if node.arity == LEAF:
        node.coeffs = nn.Parameter(node.coeffs.to(device))
    if node.arity == UNARY:
        node.weight = nn.Parammeter(node.weight.to(device))
        node.bias = nn.Parameter(node.bias.to(device))

    # Move children
    for child in node.children:
        move_node_to_device(child, device)

    return node

class ExpressionTree(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.to(DEVICE)

    def to(self, device):
        self.root = move_node_to_device(self.root, device)
        return self

    def forward(self, x):
        return eval_tree(self.root, x)
    
    def to_str(self):
        return node_to_str(self.root)
    
    def ordered_print(self):
        ordered_print(self.root)

    def to_sequence(self):
        return to_sequence(self.root)

def get_parameters(node):
    params = []

    # If this node has parameters, add them
    if node.arity == LEAF:
        params.append(node.coeffs)
    if node.arity == UNARY:
        params.append(node.weight)
        params.append(node.bias)

    # Recurse on all children and extend
    for child in node.children:
        params.extend(get_parameters(child))

    return params

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
    return eval_node(tree.root,x)


def level_order_nodes(node):
    nodes = []
    q = deque([node])

    while q:
        node = q.popleft()
        nodes.append(node)
        for child in node.children:
            q.append(child)
    return nodes