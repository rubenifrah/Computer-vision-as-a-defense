import torch
import torch.nn as nn
import kornia

class MeanShift(nn.Module):
    def __init__(self, bandwidth=0.1, num_iterations=5):
        super().__init__()
        self.bandwidth = bandwidth
        self.num_iterations = num_iterations

    def forward(self, x):
        # Kornia mean_shift expects (B, C, H, W). It returns the shifted image
        if x.ndim == 3: x = x.unsqueeze(0)
        out = kornia.filters.mean_shift(
            x, 
            bandwidth=self.bandwidth, 
            num_iterations=self.num_iterations
        )
        return out
