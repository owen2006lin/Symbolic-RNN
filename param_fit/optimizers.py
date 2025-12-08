import torch
from expr.ops import *
from expr.tree import *

#given a certain tree, how to optimize the coefficients to reduce nrsme?
#Let targets be the proper y values, and the inputs be the vectors input into the tree
#to ouput these values. Remember that the tree maps R^n -> R meaning we need multiple vectors
#so inputs expects to have shape (num_samples, n)
def MSE(tree, inputs, targets):
    n = targets.shape[0]
    preds = torch.empty(n).to(DEVICE)
    for i in range(n):
        preds[i] = eval_tree(tree, inputs[i])
    mse = torch.mean((preds - targets) ** 2)
    return mse


def ADAM_OPTIMIZE(tree, inputs, targets, steps = 2000, lr = 3e-3):
    params = get_parameters(tree.root)
    optimizer = torch.optim.Adam(params, lr = lr)
    
    for i in range(steps):
        optimizer.zero_grad()
        loss = MSE(tree, inputs, targets)
        loss.backward()
        optimizer.step()

    

def LBFGS_OPTIMIZE(tree, inputs, targets, max_iter = 50, tol_grad = 1e-7, tol_change = 1e-9):
    params = get_parameters(tree.root)
    lbfgs = torch.optim.LBFGS(params, lr = 1.0, max_iter = max_iter, tolerance_grad = tol_grad, tolerance_change = tol_change, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = MSE(tree, inputs, targets)
        loss.backward()
        return loss
    
    lbfgs.step(closure)


def Reward(tree, inputs, targets):
    mse = MSE(tree,inputs, targets)
    return 1/(1+mse)