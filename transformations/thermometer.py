import torch
import torch.nn as nn
import torch.autograd as autograd

class ThermometerFunction(autograd.Function):
    """
    implement thermometer encoding with Straight-Through Estimator (STE) from "THERMOMETER ENCODING: ONE HOT WAY TO RESIST
ADVERSARIAL EXAMPLES" by Jacob Buckman et al.

    """
    @staticmethod
    def forward(ctx, input_tensor, thresholds):
        # input_tensor: (B, C, H, W)
        # thresholds: (Levels)
        
        # Input becomes (B, C, 1, H, W)
        x_expanded = input_tensor.unsqueeze(2)
        
        # Thresholds becomes (1, 1, Levels, 1, 1)
        t_expanded = thresholds.view(1, 1, -1, 1, 1)
        
        # Thermometer encoding : 1 if pixel > threshold, else 0
        # Output shape: (B, C, Levels, H, W)
        output = (x_expanded > t_expanded).float()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output shape: (B, C, Levels, H, W)
        # we sum over the Levels dimension to propagate gradients
        grad_input = grad_output.sum(dim=2)
        
        return grad_input, None

class ThermometerEncoding(nn.Module):
    """
    Thermometer Encoding Layer:
    simply encodes each pixel using thermometer encoding with specified number of levels.
    """
    def __init__(self, levels=10):
        super().__init__()
        self.levels = levels
        self.register_buffer('thresholds', torch.linspace(0, 1, steps=levels+1)[1:])

    def forward(self, x):
        encoded = ThermometerFunction.apply(x, self.thresholds)
        
        B, C, L, H, W = encoded.shape
        return encoded.view(B, C * L, H, W)