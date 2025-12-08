import torch
from expr.tree import *
from expr.ops import *
from .optimizers import * 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
This first section will be on testing computing MSE of the tree. Feel free to comment out if only care about testing
other components
'''


#First create some leaf nodes
leaf1 = Node.newLeaf(UNARY_OP["id"], torch.tensor([1.0, 2.0]), LEAF)
leaf2 = Node.newLeaf(UNARY_OP["id"], torch.tensor([0.5, -1.0]), LEAF)
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
inputs = inputs.to(DEVICE)

#The targets are already known.
targets = torch.tensor([
    (1*1 + 2*2) + (0.5*1 + -1*2),   # 5 + (-1.5) = 3.5
    (1*0 + 2*1) + (0.5*0 + -1*1),   # 2 + (-1) = 1
    (1*3 + 2*-1) + (0.5*3 + -1*-1), # 1 + 2.5 = 3.5
])
targets = targets.to(DEVICE)

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
inputs = inputs.to(DEVICE)
x = inputs
#First case will include a small bit of error by changing the numbers
targets_error = (
    torch.dot(torch.exp(x[0]), torch.tensor([1.0, 2.0, 3.0], device = DEVICE)) +
    torch.dot(x[0], torch.tensor([4.0, 3.0, 2.1], device = DEVICE)) +    #2.1 instead of 2.0 to simulate just a bit of error
    torch.dot(torch.sin(x[0]), torch.tensor([1.1, 3.4, 2.1], device = DEVICE)),  #Again 1.1 instead of 1.2 to simulate some error
    
    torch.dot(torch.exp(x[1]), torch.tensor([1.0, 2.0, 2.95], device = DEVICE)) + #2.95 instead of 3.0
    torch.dot(x[1], torch.tensor([4.0, 3.0, 2.6], device = DEVICE)) + #2.6 instead of 2.0
    torch.dot(torch.sin(x[1]), torch.tensor([1.2, 3.4, 2.1], device = DEVICE))
)
targets_error = torch.tensor(targets_error)
targets_error = targets_error.to(DEVICE)

mse_value = MSE(tree, inputs, targets_error)
print("Test 2: MSE:", mse_value.item())


#Final case will be exact, expecting 0 error
targets_exact = (
    torch.dot(torch.exp(x[0]), torch.tensor([1.0, 2.0, 3.0], device = DEVICE)) +
    torch.dot(x[0], torch.tensor([4.0, 3.0, 2.0], device = DEVICE)) +
    torch.dot(torch.sin(x[0]), torch.tensor([1.2, 3.4, 2.1], device = DEVICE)),
    
    torch.dot(torch.exp(x[1]), torch.tensor([1.0, 2.0, 3.0], device = DEVICE)) +
    torch.dot(x[1], torch.tensor([4.0, 3.0, 2.0], device = DEVICE)) +
    torch.dot(torch.sin(x[1]), torch.tensor([1.2, 3.4, 2.1], device = DEVICE))
)
targets_exact = torch.tensor(targets_exact)
targets_exact = targets_exact.to(DEVICE)

mse_value = MSE(tree, inputs, targets_exact)
print("Test 3: MSE:", mse_value.item())
'''


This next section tests out the ADAM and LBFGS optimizers of the tree. Feel free to comment out if testing other components
'''

torch.set_default_dtype(torch.float64)

root = Node.newNode(UNARY_OP["id"], ROOT)
mid = Node.newNode(BINARY_OP["add"], BINARY)
root.add_child(mid)
root.weight.requires_grad = False
root.bias.requires_grad = False


leaf1 = Node.newLeaf(UNARY_OP["exp"], torch.tensor([1.0,1.0,1.0]), LEAF)
leaf2 = Node.newLeaf(UNARY_OP["id"], torch.tensor([1.0, 1.0, 1.0]), LEAF)
leaf3 = Node.newLeaf(UNARY_OP["sin"], torch.tensor([1.0,1.0, 1.0]), LEAF)


mid.add_child(leaf1)
mid.add_child(leaf2)
mid.add_child(leaf3)

tree = ExpressionTree(root)


inputs = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.1, 0.2, 0.4],
    [0.3, 1.0, 0.7],
    [1.5, 1.1, 0.3],
    [2.0, 0.4, 1.3],
    [2.3, 1.7, 0.2],
    [0.9, 2.1, 1.4],
], dtype=torch.float64)

a_true = torch.tensor([1.0, 2.0, 3.0])
b_true = torch.tensor([4.0, 3.0, 2.0])
c_true = torch.tensor([1.2, 3.4, 2.1])

targets_exact = []
for x in inputs:
    y = (
        torch.dot(torch.exp(x), a_true)
        + torch.dot(x, b_true)
        + torch.dot(torch.sin(x), c_true)
    )
    targets_exact.append(y)

targets_exact = torch.stack(targets_exact)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tree.to(DEVICE)
inputs = inputs.to(DEVICE)
targets_exact = targets_exact.to(DEVICE)

#in this case here, we've set all the coefficients to 1, and all the targets to the ones from the above test
#ADAM should chang ethe coefficients to be as close as possible to the default tree above

print("Original Tree: ")
print(tree.to_sequence())
print(tree.to_str())
tree.ordered_print()
print(f"MSE: {MSE(tree,inputs, targets_exact).item()}")
print("\n")

ADAM_OPTIMIZE(tree,inputs, targets_exact)

print("ADAM_OPTIMIZED Tree: ")
print(tree.to_sequence())
print(tree.to_str())
tree.ordered_print()
print(f"MSE: {MSE(tree,inputs, targets_exact).item()}")
print("\n")

LBFGS_OPTIMIZE(tree, inputs, targets_exact)

print("LBFGS Tree: ")
print(tree.to_sequence())
print(tree.to_str())
tree.ordered_print()
print(f"MSE: {MSE(tree,inputs, targets_exact).item()}")
print("\n")

print(mid.children[0].coeffs)
print(mid.children[1].coeffs)
print(mid.children[2].coeffs)


#Something interesting to note is that the output is not EXACTLY equal to the expected coefficients
#This is because the we can see the error is extremely small O ~ 10^-8, which means the rate at which
#our loss moves is much smaller than our coefficients so the gradient gets very small. You might also
#notice the optimizers are architecture dependent, and you might get better results utilizing CUDA cores
#than on cpu



