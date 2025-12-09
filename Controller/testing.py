import torch
from .controller import *
from expr.tree import *
from expr.ops import *
from param_fit.optimizers import *
from rl.agent import *
torch.autograd.set_detect_anomaly(True)

def generate_test_data(
    n_samples=512,
    noise_std=0.0,
    device="cpu",
):
    """
    Generate synthetic (X, y) for the tree:

        root:  id
          |
         add
      /   |    \
    exp   id   sin

    Leaf semantics: for a sample x (shape [3])
        leaf1 = <exp(x), a>
        leaf2 = <x,       b>
        leaf3 = <sin(x),  c>
        y = leaf1 + leaf2 + leaf3

    where a, b, c are fixed coefficient vectors.
    """

    # Sample inputs
    X = torch.empty(n_samples, 3, device=device).uniform_(-2.0, 2.0)

    # Coefficients for the three leaves (choose whatever you like)
    a = torch.tensor([1.0, 2.0, 3.0], device=device)   # for exp(x)
    b = torch.tensor([4.0, -3.0, 2.0], device=device)  # for id(x)
    c = torch.tensor([1.2, -0.7, 0.5], device=device)  # for sin(x)

    # Elementwise ops then dot with coeffs
    exp_term = (torch.exp(X) * a).sum(dim=1)      # <exp(X), a>
    id_term  = (X * b).sum(dim=1)                 # <X, b>
    sin_term = (torch.sin(X) * c).sum(dim=1)      # <sin(X), c>

    y = exp_term + id_term + sin_term            # [n_samples]

    if noise_std > 0.0:
        y = y + noise_std * torch.randn_like(y)

    return X, y


def build_ground_truth_tree():
    # Root: unary id
    root = Node.newNode(UNARY_OP["id"], ROOT)
    mid  = Node.newNode(BINARY_OP["add"], BINARY)
    root.add_child(mid)

    # Freeze root weight/bias if you use them
    root.weight.requires_grad = False
    root.bias.requires_grad = False

    # Leaf 1: exp( 1*x1 + 2*x2 + 3*x3 )
    leaf1 = Node.newLeaf(
        UNARY_OP["exp"],
        torch.tensor([1.0, 2.0, 3.0]),
        LEAF,
    )

    # Leaf 2: id( 4*x1 - 3*x2 + 2*x3 ) == 4*x1 - 3*x2 + 2*x3
    leaf2 = Node.newLeaf(
        UNARY_OP["id"],
        torch.tensor([4.0, -3.0, 2.0]),
        LEAF,
    )

    # Leaf 3: sin( 1.2*x1 - 0.7*x2 + 0.5*x3 )
    leaf3 = Node.newLeaf(
        UNARY_OP["sin"],
        torch.tensor([1.2, -0.7, 0.5]),
        LEAF,
    )

    mid.add_child(leaf1)
    mid.add_child(leaf2)
    mid.add_child(leaf3)

    return root 

def build_ground_truth_expression_tree():
    root = build_ground_truth_tree()
    return ExpressionTree(root)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X, y_true = generate_test_data(128, noise_std=0.0, device=DEVICE)
tree = build_ground_truth_expression_tree().to(DEVICE)
'''
with torch.no_grad():
    y_tree_list = []
    for i in range(X.shape[0]):
        y_tree_list.append(eval_tree(tree, X[i]))
    y_tree = torch.stack(y_tree_list)

print(torch.max(torch.abs(y_true - y_tree)))
tree.ordered_print()
print(tree.to_str())
'''


policy = TreePolicy(5, RL_VOCAB_SIZE)
policy = policy.to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr = 3e-4)
pool = []


for t in range(100):
    stats, pool = rl_training_step(
        policy,
        optimizer,
        X,
        y_true,
        50,
        5,
        4,
        0.2,
        DEVICE,
        pool,
        4
    )
    print (t, stats)
    for tuple in pool:
        print(tuple[1].to_str())


