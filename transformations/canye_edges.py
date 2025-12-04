import torch
import torch.nn as nn
import kornia

class StraightThrough(torch.autograd.Function):
    _call_count = 0 
    @staticmethod
    def forward(ctx, input_tensor, output_tensor):
        ctx.save_for_backward(input_tensor)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        if StraightThrough._call_count % 2 == 0:
            fill_value = 1.0
        else:
            fill_value = -1.0
        StraightThrough._call_count += 1
        grad_input = torch.full_like(input_tensor, fill_value)
        
        return grad_input, None

class CannyEdge(nn.Module):
    def __init__(self, low_threshold=100.0, high_threshold=200.0, blur=True):
        super().__init__()
        self.low = low_threshold / 255.0 if low_threshold > 1.0 else low_threshold
        self.high = high_threshold / 255.0 if high_threshold > 1.0 else high_threshold
        self.blur = blur

    def forward(self, x):
        original_x = x
        
        if x.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)
        if self.blur:
            x = kornia.filters.gaussian_blur2d(x, (5, 5), (1.5, 1.5))

        _, edges = kornia.filters.canny(
            x, 
            low_threshold=self.low, 
            high_threshold=self.high, 
            kernel_size=(5, 5), 
            hysteresis=True
        )
        return StraightThrough.apply(original_x, edges)