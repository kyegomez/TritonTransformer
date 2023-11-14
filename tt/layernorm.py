import torch
import triton
import triton.language as tl
from torch import nn

@triton.jit
def layer_norm_kernel(
    x,
    mean,
    var,
    gamma,
    beta,
    epsilon,
    stride_xm,
    stride_xn,
    stride_gamma,
    stide_beta,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for layer normalization.

    Parameters:
    x - Input tensor.
    mean - Tensor to store computed means.
    var - Tensor to store computed variances.
    gamma - Scale tensor.
    beta - Shift tensor.
    epsilon - A small value to avoid division by zero.
    stride_xm, stride_xn - Strides for the input tensor.
    stride_gamma, stride_beta - Strides for Gamma and Beta tensors.
    n - Size of the last dimension of the input tensor.
    BLOCK_SIZE - Size of the block for Triton computation.
    """
    # Compute indices for this thread
    row = tl.program_id(0)

    # Compute memory offsets
    x_ptrs = x + row * stride_xm
    mean_ptrs = mean + row
    var_ptrs = var + row
    gamma_ptrs = gamma
    beta_ptrs = beta

    # Load and compute mean
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n, other=0)
    mean = tl.sum(x, axis=0) / n
    tl.store(mean_ptrs, mean)

    # Load and compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n
    tl.store(var_ptrs, var)

    # Normalize
    std = tl.sqrt(var + epsilon)
    y = (x_centered / std) * tl.load(
        gamma_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n, other=1
    ) + tl.load(beta_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n, other=0)

    # Store result
    tl.store(x_ptrs, y, mask=tl.arange(0, BLOCK_SIZE) < n)


class TritonLayerNorm(nn.Module):
    """
    Initializes the Triton-based layer normalization module.

    Parameters:
    normalized_shape - The shape of the input tensor.
    eps - A small value to avoid division by zero during normalization.
    block_size - The size of the block to be processed by each thread.
    """

    def __init__(self, norm_shape, eps=1e-5, BLOCK_SIZE=128):
        super(TritonLayerNorm, self).__init__()
        self.norm_shape = norm_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta = nn.Parameter(torch.ones(norm_shape))
        self.block_size = BLOCK_SIZE

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the layer normalization.

        Parameters:
        x - Input tensor of any shape.

        Returns:
        Normalized tensor with the same shape as input.
        """
        orig_shape = x.shape
        x_reshaped = x.reshape(-1, self.norm_shape)

        # Allocate memory for intermedate computations
        mean = torch.empty(x_reshaped.shape[0], device=x.device)
        var = torch.empty_like(mean)

        # Calculate grid size for triton kernel
        grid = (x_reshaped.shape[0],)

        # Invoke Triton kernel
        layer_norm_kernel(
            x_reshaped,
            mean,
            var,
            self.gamma,
            self.beta,
            self.eps,
            x_reshaped.stride(0),
            x_reshaped.stride(1),
            self.gamma.stride(0),
            self.beta.stride(0),
            self.norm_shape,
            self.block_size,
        )

        # Reshape back to original shape
        return x_reshaped.reshape(orig_shape)
    