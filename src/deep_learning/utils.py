import numpy as np
from typing import Any, Tuple
import numpy.typing as npt


def ensure_tensor(obj: Any):
    """Ensure that the input is a Tensor object, raises TypeError if not."""
    from deep_learning.tensor import Tensor
    
    if not isinstance(obj, Tensor):
        raise TypeError(f"Expected a Tensor, got {type(obj)} instead.")

    return obj


def unbroadcast(grad: npt.NDArray, shape: Tuple[int, ...]) -> npt.NDArray:
    """
    Reduce gradient to original shape by summing over broadcasted dimensions.
    This handles the inverse of NumPy broadcasting.
    """
    # Remove extra leading dimensions
    ndims_added = len(grad.shape) - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    
    # Sum over broadcasted dimensions (size 1 -> size N)
    for i, (dim, grad_dim) in enumerate(zip(shape, grad.shape)):
        if dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


def correlate2d(input: np.ndarray, kernel: np.ndarray):
    """
    Perform a 2D correlation (cross-correlation) between a single 2D input and kernel.
    Simplified version with stride=1, dilation=1.
    """
    H_in, W_in = input.shape
    kH, kW = kernel.shape

    # Output dimensions
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    output = np.zeros((H_out, W_out), dtype=input.dtype)

    for i in range(H_out):
        for j in range(W_out):
            # Extract region from input
            region = input[i:i + kH, j:j + kW]
            output[i, j] = np.sum(np.multiply(region, kernel))

    return output


def convolve2d(input: np.ndarray, kernel: np.ndarray):
    """Performs 2D convolution by flipping the kernel and calling correlate2d."""
    return correlate2d(input, np.flip(kernel))
