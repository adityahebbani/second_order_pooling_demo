import torch
import torch.nn as nn

class SecondOrderPooling(nn.Module):
    def __init__(self):
        super(SecondOrderPooling, self).__init__()

    def forward(self, x):
        """
        Perform second-order pooling on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Covariance representation of the input.
        """
        # Input dimensions
        batch_size, channels, height, width = x.size()

        # Reshape to (batch_size, channels, height * width)
        x = x.view(batch_size, channels, -1)

        # Subtract the mean along the spatial dimensions
        x_centered = x - x.mean(dim=2, keepdim=True)

        # Compute the covariance matrix: (batch_size, channels, channels)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (height * width)

        # Flatten the covariance matrices
        cov_flat = cov.view(batch_size, -1)

        return cov_flat
