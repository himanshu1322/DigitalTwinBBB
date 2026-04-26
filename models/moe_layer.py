import torch
import torch.nn as nn
import torch.nn.functional as F
from .bit_linear import BitLinear158

class BBBMoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k # Number of experts to activate per molecule (k=2 is better than k=1)
        
        # The Router: Decides which experts are best
        self.router = nn.Linear(dim, num_experts)
        
        # The Experts: Specialized 1.58-bit brains
        self.experts = nn.ModuleList([BitLinear158(dim, dim) for _ in range(num_experts)])
        
        # Batch Normalization: Helps 1.58-bit layers stabilize their weights
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # 1. Routing Scores
        # We add a tiny bit of noise during training to force expert exploration
        logits = self.router(x)
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.1
            
        # 2. Top-K Selection
        # We pick the 'K' best experts and weight their contributions
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        # Re-normalize scores so they sum to 1
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # 3. Expert Aggregation
        # We process each expert and multiply by the router's confidence score
        batch_size = x.size(0)
        final_output = torch.zeros_like(x)
        
        # Loop through experts (efficient for small num_experts)
        for i, expert in enumerate(self.experts):
            # Find which batch items selected this specific expert
            mask = (top_indices == i) if 'top_indices' in locals() else (top_k_indices == i)
            if mask.any():
                # Get the row indices where this expert was chosen
                row_indices, col_indices = torch.where(mask)
                
                # Run the selected data through the expert
                expert_out = expert(x[row_indices])
                
                # Weight the expert's output by the router's score
                weight = top_k_scores[row_indices, col_indices].unsqueeze(-1)
                final_output[row_indices] += expert_out * weight
                
        return self.bn(final_output)