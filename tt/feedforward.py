# !pip install torch
# !pip install triton
import torch
import triton
import triton.language as tl


# Ensure a CUDA device is available
if not torch.cuda.is_available():
    raise RuntimeError("This model requires a CUDA-enabled GPU to run.")


# Define a reasonable block size. This should be tuned for your specific GPU.
BLOCK_SIZE = 128


# ---------------------------
# Matrix Multiplication Kernel in Triton
# ---------------------------
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_cm
):
    # Compute indices for this thread
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, BLOCK_SIZE)  # Use the global BLOCK_SIZE

    # Compute the memory offsets
    a_ptrs = A + m * stride_am + k * stride_ak
    b_ptrs = B + k * stride_bk + n * stride_bn
    c_ptrs = C + m * stride_cm + n * stride_cn

    # Perform the matrix multiplication
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)

    # Atomic addition to accumulate results
    tl.atomic_add(c_ptrs, c)


# ---------------------------
# ReLU Activation Kernel in Triton
# ---------------------------
@triton.jit
def relu_kernel(X, Y, N):
    idx = tl.program_id(0)
    if idx < N:
        x = tl.load(X + idx)
        y = tl.max(x, 0)  # ReLU activation
        tl.store(Y + idx, y)


# ---------------------------
# Triton-Based Feedforward Network
# ---------------------------
class TritonFeedForward(torch.nn.Module):
    def __init__(self, input_features: int, hidden_features: int, output_features: int):
        """
        Initializes the feedforward network.
        Parameters:
        - input_features (int): Number of input features.
        - hidden_features (int): Number of hidden features.
        - output_features (int): Number of output features.
        """
        super(TritonFeedForward, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.weights1 = torch.nn.Parameter(torch.randn(input_features, hidden_features))
        self.bias1 = torch.nn.Parameter(torch.randn(hidden_features))
        self.weights2 = torch.nn.Parameter(
            torch.randn(hidden_features, output_features)
        )
        self.bias2 = torch.nn.Parameter(torch.randn(output_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feedforward network.
        Parameters:
        - x (torch.Tensor): Input tensor with shape [batch_size, input_features].
        Returns:
        - torch.Tensor: Output tensor with shape [batch_size, output_features].
        """
        # Calculate grid dimensions for Triton kernels
        grid = (
            x.shape[0],
            self.hidden_features // BLOCK_SIZE,
            self.input_features // BLOCK_SIZE,
        )

        # First Linear Layer
        hidden = matmul_kernel[grid](
            x,
            self.weights1,
            self.bias1,
            x.shape[0],
            self.hidden_features,
            self.input_features,
            x.stride(0),
            x.stride(1),
            self.weights1.stride(0),
            self.weights1.stride(1),
            self.bias1.stride(0),
            BLOCK_SIZE,
        )  # Corrected stride access for bias1
        # ReLU Activation
        hidden = relu_kernel[grid](hidden, hidden.shape[0])
        # Second Linear Layer
        output = matmul_kernel[grid](
            hidden,
            self.weights2,
            self.bias2,
            hidden.shape[0],
            self.output_features,
            self.hidden_features,
            hidden.stride(0),
            hidden.stride(1),
            self.weights2.stride(0),
            self.weights2.stride(1),
            self.bias2.stride(0),
            BLOCK_SIZE,
        )  # Corrected stride access for bias2
        return output


device = torch.device("cuda")  # Define a PyTorch device targeting the GPU

# ---------------------------
# Example Usage
# ---------------------------
# Define input, hidden, and output features
input_features = 128
hidden_features = 256
output_features = 128
batch_size = 32

# Initialize the feedforward network
feedforward = TritonFeedForward(input_features, hidden_features, output_features)

# Example input tensor
input_tensor = torch.randn(batch_size, input_features).to(device)

# Forward pass
output = feedforward(input_tensor)
print(f"Output Shape: {output}")
