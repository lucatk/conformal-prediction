import torch
import torch.nn as nn
import torch.nn.functional as F


class UnimodalHead(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, hidden_dim=128):
        super(UnimodalHead, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)  # Intermediate hidden layer
        self.output = nn.Linear(hidden_dim, num_classes)  # Produces η(x)
        self.num_classes = num_classes

    def forward(self, x):
        # Step 1: Compute η(x) = f(x; θ)
        eta = self.output(F.relu(self.hidden(x)))  # η ∈ ℝ^K — raw logits for each class

        # Step 2: Ensure all values are non-negative: v_k = φ(η_k), φ = softplus ensures v_k ≥ 0
        v = F.softplus(eta)  # v ∈ ℝ_+^K

        # Step 3: Generate cumulative sum: r_k = r_{k-1} + v_k (with r₁ = v₁)
        r = torch.cumsum(v, dim=1)  # r is non-decreasing over k

        # Step 4: Apply symmetric decreasing function: z_k = ψ_E(r_k) = -|r_k|
        z = -torch.abs(r)  # z ∈ ℝ^K, unimodal due to symmetric log-probability decay

        # Step 5: Compute class probabilities: p̂_k = softmax(z_k)
        probs = F.softmax(z, dim=1)  # p̂ ∈ Δ^K (K-dimensional simplex, unimodal distribution)

        return probs
