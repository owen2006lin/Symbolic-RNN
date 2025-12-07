import torch
from expr.ops import *
from expr.tree import *

#given a certain tree, how to optimize the coefficients to reduce nrsme?
#Let targets be the proper y values, and the inputs be the vectors input into the tree
#to ouput these values. Remember that the tree maps R^n -> R meaning we need multiple vectors
#so inputs expects to have shape (num_samples, n)
def MSE(tree, inputs, targets):
    n = targets.shape[0]
    preds = torch.empty(n)
    for i in range(n):
        preds[i] = eval_tree(tree, inputs[i])
    mse = torch.mean((preds - targets) ** 2)
    return mse





