import torch
from expr.tree import *
from expr.ops import *
from .optimizers import MSE 

#First create some leaf nodes
leaf1 = Node(func=UNARY_OP["id"], coeffs=torch.tensor([1.0, 2.0]), arity=LEAF)
leaf2 = Node(func=UNARY_OP["id"], coeffs=torch.tensor([0.5, -1.0]), arity=LEAF)
#Root node w addition operator
mid = Node.newNode(BINARY_OP["add"], arity=BINARY)
mid.add_child(leaf1)
mid.add_child(leaf2)

root = Node.newNode(UNARY_OP["id"], ROOT)
root.add_child(mid)

tree = ExpressionTree(root)

#input is 2d tensor
inputs = torch.tensor([
    [1.0, 2.0],
    [0.0, 1.0],
    [3.0, -1.0],
])

#The targets are already known.
targets = torch.tensor([
    (1*1 + 2*2) + (0.5*1 + -1*2),   # 5 + (-1.5) = 3.5
    (1*0 + 2*1) + (0.5*0 + -1*1),   # 2 + (-1) = 1
    (1*3 + 2*-1) + (0.5*3 + -1*-1), # 1 + 2.5 = 3.5
])


mse_value = MSE(tree, inputs, targets)
print("Test 1: MSE:", mse_value.item())


#Next test case, this time with the example tree from the paper
root = Node.newNode(UNARY_OP["id"], ROOT)
mid = Node.newNode(BINARY_OP["add"], BINARY)
root.add_child(mid)

leaf1 = Node.newLeaf(UNARY_OP["exp"], torch.tensor([1.0,2.0,3.0]), LEAF)
leaf2 = Node.newLeaf(UNARY_OP["id"], torch.tensor([4.0, 3.0, 2.0]), LEAF)
leaf3 = Node.newLeaf(UNARY_OP["sin"], torch.tensor([1.2,3.4, 2.1]), LEAF)

mid.add_child(leaf1)
mid.add_child(leaf2)
mid.add_child(leaf3)

tree = ExpressionTree(root)

inputs = torch.tensor([
    [1.2, 3.4, 5.6],
    [5.5, 1.2, 4.6]
])

x = inputs
#First case will include a small bit of error by changing the numbers
targets_error = (
    torch.dot(torch.exp(x[0]), torch.tensor([1.0, 2.0, 3.0])) +
    torch.dot(x[0], torch.tensor([4.0, 3.0, 2.1])) +    #2.1 instead of 2.0 to simulate just a bit of error
    torch.dot(torch.sin(x[0]), torch.tensor([1.1, 3.4, 2.1])),  #Again 1.1 instead of 1.2 to simulate some error
    
    torch.dot(torch.exp(x[1]), torch.tensor([1.0, 2.0, 2.95])) + #2.95 instead of 3.0
    torch.dot(x[1], torch.tensor([4.0, 3.0, 2.6])) + #2.6 instead of 2.0
    torch.dot(torch.sin(x[1]), torch.tensor([1.2, 3.4, 2.1]))
)
targets_error = torch.tensor(targets_error)

mse_value = MSE(tree, inputs, targets_error)
print("Test 2: MSE:", mse_value.item())


#Final case will be exact, expecting 0 error
targets_exact = (
    torch.dot(torch.exp(x[0]), torch.tensor([1.0, 2.0, 3.0])) +
    torch.dot(x[0], torch.tensor([4.0, 3.0, 2.0])) +
    torch.dot(torch.sin(x[0]), torch.tensor([1.2, 3.4, 2.1])),
    
    torch.dot(torch.exp(x[1]), torch.tensor([1.0, 2.0, 3.0])) +
    torch.dot(x[1], torch.tensor([4.0, 3.0, 2.0])) +
    torch.dot(torch.sin(x[1]), torch.tensor([1.2, 3.4, 2.1]))
)
targets_exact = torch.tensor(targets_exact)

mse_value = MSE(tree, inputs, targets_exact)
print("Test 3: MSE:", mse_value.item())
