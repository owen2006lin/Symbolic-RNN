from .agent import *
from param_fit.optimizers import *
from expr.ops import *
from expr.tree import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



root = Node.newNode(UNARY_OP["id"], ROOT)
mid = Node.newNode(BINARY_OP["add"], BINARY)
root.add_child(mid)
root.weight.requires_grad = False
root.bias.requires_grad = False


leaf1 = Node.newLeaf(None, torch.tensor([1.0,1.0,1.0]), LEAF)
leaf2 = Node.newLeaf(None, torch.tensor([1.0, 1.0, 1.0]), LEAF)
leaf3 = Node.newLeaf(None, torch.tensor([1.0,1.0, 1.0]), LEAF)


mid.add_child(leaf1)
mid.add_child(leaf2)
mid.add_child(leaf3)

tree = ExpressionTree(root)

max_nodes = 5
policy = TreePolicy(max_nodes = max_nodes, rl_vocab_size= RL_VOCAB_SIZE).to(DEVICE)
root, log_prob = sample_ops_for_tree(policy, root, RL_OPS, UNARY_MASK, BINARY_MASK, device = DEVICE)

tree = ExpressionTree(root)
tree.ordered_print()