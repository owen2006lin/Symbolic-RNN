import numpy as np
import function as func
import torch
from enum import Enum, auto


class nodeType(Enum):
    #Helpful to distinguish when in the middle of the tree
    UNARY = auto()
    BINARY = auto()
    #Leaves + roots not necessary (unary is implicit)
    ROOT_UNARY = auto()
    LEAF_UNARY = auto()

class Node:
    def __init__(self, func = None, coeffs = None, weight = None, bias = None, arity = None):
        self.op = func
        self.children = []
        #For leaves: coefficients
        self.coeffs = coeffs if coeffs is not None else torch.empty(0) #Torch tensor type
        #weights and biases for non leaf unaries
        self.weight = 1
        self.bias = 0
        self.arity = arity
    
    def add_child(self, node : "Node"):
        self.children.append(node)


class ExpressionTree:
    def __init__(self, root):
        self.root = root
        self.leaves = None
        self.sequence = None
        self.width = 0
        self.depth = 0

#building simplest example tree, unary root and binary operator with n children as described on the second page of paper
def identity(n):
    return n

def build_default_tree():
    coeffs = torch.tensor([1], dtype = float)
    root = Node(identity, arity = nodeType.ROOT_UNARY)
    binary = Node(torch.add, arity = nodeType.BINARY)
    root.add_child(binary)

    left = Node(torch.exp, coeffs, arity =  nodeType.LEAF_UNARY)
    mid = Node(identity, coeffs, arity = nodeType.LEAF_UNARY)
    right = Node(torch.sin, coeffs, arity = nodeType.LEAF_UNARY)

    binary.add_child(left)
    binary.add_child(mid)
    binary.add_child(right)
    
    tree =  ExpressionTree(root)
    tree.leaves = 3
    return tree

#input x should be numpy tensor type, but on recursive calls it becomes scalar
def eval_node(node,x):
    if node.arity == nodeType.LEAF_UNARY:
        return torch.dot(node.op(x), node.coeffs)
    if not node.children:
        return(node.op(x))
    if node.arity == nodeType.UNARY or node.arity == nodeType.ROOT_UNARY:
        return(node.weight*node.op(eval_node(node.children[0],x)) + node.bias)
    if node.arity == nodeType.BINARY:
        values = [eval_node(c,x) for c in node.children]
        result = values[0]
        for v in values[1:]:
            result = node.op(result, v)
        return result


def eval_tree(tree,x):
    return eval_node(tree.root,x).item()





basictree = build_default_tree()
input = torch.tensor([1], dtype = float)
print(eval_tree(basictree, input))