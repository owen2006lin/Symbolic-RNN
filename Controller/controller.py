import torch
from expr.tree import *
from expr.ops import *
from param_fit.optimizers import *
from rl.agent import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_default_skeleton():
    root = Node.newNode(None, ROOT)
    mid = Node.newNode(None, BINARY)

    root.add_child(mid)
    default_coeffs = torch.tensor([1,1,1], dtype = torch.float32, device = DEVICE)
    leaf1 = Node.newLeaf(None, default_coeffs, LEAF)
    leaf2 = Node.newLeaf(None, default_coeffs, LEAF)
    leaf3 = Node.newLeaf(None, default_coeffs, LEAF)

    mid.add_child(leaf1)
    mid.add_child(leaf2)
    mid.add_child(leaf3)
    tree = ExpressionTree(root)
    return tree

BAD_REWARD = 0.0
def sample_and_eval_tree(policy, inputs, targets, T1, T2, device):
    tree = build_default_skeleton()
    root = tree.root
    root, total_log_prob = sample_ops_for_tree(
        policy,
        root,
        RL_OPS,
        UNARY_MASK,
        BINARY_MASK,
        device
    )
    try:
        ADAM_OPTIMIZE(tree, inputs, targets, T1)
        LBFGS_OPTIMIZE(tree, inputs, targets, T2)
        with torch.no_grad():
            reward = Reward(tree, inputs, targets)
            reward = torch.tensor(float(reward), device = device)
    
    except RuntimeError as e:
        print("[sample_and_eval_tree] Numerical failure for tree: ", e)
        reward = torch.tensor(BAD_REWARD, device = device, dtype = torch.float32)

    return reward, total_log_prob, tree


def rl_training_step(
    policy, 
    optimizer,
    inputs,
    targets,
    T1,
    T2,
    N,
    alpha = 0.2,
    device = "cuda",
    pool = None,
    K = None
):
    """
    Perform one RL update step on the policy:
      - sample N trees,
      - coarse-tune + get reward,
      - compute risk-seeking policy gradient,
      - optimizer.step().
    """
    if pool is None:
        pool = []
    
    policy.train()
    rewards = []
    log_probs = []
    trees = []

    for _ in range (N):
        R, log_p, tree = sample_and_eval_tree(policy, inputs, targets, T1, T2, device)
        rewards.append(R)
        log_probs.append(log_p)
        trees.append(tree)
    rewards = torch.stack(rewards)       
    log_probs = torch.stack(log_probs)
    
    R_hat_alpha = torch.quantile(rewards.detach(), q= 1.0 - alpha)

    indicators = (rewards >= R_hat_alpha).float()
    advantages = (rewards - R_hat_alpha)*indicators
    
    loss = - (advantages.detach() * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for R, tree in zip(rewards, trees):
        pool.append((float(R.item()), tree))
    pool.sort(key = lambda rt : rt[0], reverse = True)
    pool = pool[:K]

    stats = {
        "loss": loss.item(),
        "mean_reward": rewards.mean().item(),
        "max_reward": rewards.max().item(),
        "R_hat_alpha": R_hat_alpha.item(),
    }

    return stats, pool