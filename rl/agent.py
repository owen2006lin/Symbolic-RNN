from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from expr.ops import *
from expr.tree import *



class TreePolicy(nn.Module):
    def __init__(
        self, 
        max_nodes: int, 
        rl_vocab_size: int, 
        hidden_size: int = 128, 
        node_embed_dim: int= 64, 
        op_embed_dim: int = 64
): 
        super().__init__()
        self.max_nodes = max_nodes
        self.rl_vocab_size = rl_vocab_size

        self.start_op_id = rl_vocab_size

        self.node_embed = nn.Embedding(max_nodes,node_embed_dim)
        self.op_embed = nn.Embedding(rl_vocab_size + 1, op_embed_dim)


        self.gru = nn.GRUCell(node_embed_dim + op_embed_dim, hidden_size)
        self.head = nn.Linear(hidden_size, rl_vocab_size)
    
    def init_hidden(self, batch_size: int = 1, device = None):
        return torch.zeros(batch_size, self.gru.hidden_size, device = device)
    
    def forward_step(self, node_idx, prev_op_id, h_prev):
        node_e = self.node_embed(node_idx)
        op_e = self.op_embed(prev_op_id)
        x = torch.cat([node_e, op_e], dim=-1)
        h = self.gru(x, h_prev)
        logits = self.head(h)
        return logits, h

    
    
def sample_ops_for_tree(
    policy, 
    root, 
    RL_OPS,
    UNARY_MASK,
    BINARY_MASK,
    device
): 
    
    """
    Fills in node.op for each internal node of the fixed skeleton using the policy,
    traversing the tree in level order.

    Args:
        policy: TreePolicy instance
        root:   root Node of your fixed skeleton (children already wired)
        RL_OPS: list of (op_type, op_name), e.g. ("unary", "sin")
        UNARY_MASK:  BoolTensor [V], True where op is valid for unary nodes
        BINARY_MASK: BoolTensor [V], True where op is valid for binary nodes
        device: torch device; if None, inferred from policy

    Returns:
        root (with .op fields set),
        total_log_prob: scalar tensor, sum of log π(a_t | history) over all ops
    """

    nodes = level_order_nodes(root)
    log_probs = []

    h = policy.init_hidden(batch_size = 1, device = device)
    prev_op_id = torch.tensor([policy.start_op_id], dtype = torch.long, device = device)

    unary_mask = UNARY_MASK.to(device)
    binary_mask = BINARY_MASK.to(device)

    for idx, node in enumerate(nodes):

        node_idx_tensor = torch.tensor([idx], dtype = torch.long, device = device)
        logits, h = policy.forward_step(node_idx_tensor, prev_op_id, h)

        if node.arity in (UNARY, LEAF, ROOT):
            mask = unary_mask
        
        if node.arity == BINARY:
            mask = binary_mask
        
        masked_logits = logits.clone()
        masked_logits[:, ~mask] = float('-inf')
        
        dist = Categorical(logits = masked_logits)
        action = dist.sample()
        log_p = dist.log_prob(action)

        a_id = action.item()
        op_type, op_name = RL_OPS[a_id]

        if op_type == "unary":
            node.op = UNARY_OP[op_name]
        elif op_type == "binary":
            node.op = BINARY_OP[op_name]

        log_probs.append(log_p)
        prev_op_id = action

    return root, torch.stack(log_probs).sum()
