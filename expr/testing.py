from .tree import *
from .ops import *
import torch

torch.set_default_dtype(torch.float32)

def build_default_tree():
    root = Node.newNode(UNARY_OP["id"], ROOT)
    binary = Node.newNode(BINARY_OP["add"], BINARY)
    root.add_child(binary)

    left = Node.newNode(UNARY_OP["exp"], LEAF)
    mid = Node.newNode(UNARY_OP["id"], LEAF)
    right = Node.newNode(UNARY_OP["sin"], LEAF)

    binary.add_child(left)
    binary.add_child(mid)
    binary.add_child(right)
    
    tree =  ExpressionTree(root)
    tree.leaves = 3
    return tree


basictree = build_default_tree()
input = torch.tensor([1], dtype = torch.float32)
print(eval_tree(basictree, input))