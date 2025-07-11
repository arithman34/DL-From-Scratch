import numpy as np
from typing import Optional, Union, Tuple
import numpy.typing as npt

from deep_learning.backend import EPSILON
from deep_learning.utils import correlate2d, convolve2d

Number = Union[int, float]


def ensure_tensor(obj):
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


# Arithmetic operations
def add(input_a, input_b):
    """Add two tensors element-wise."""
    from deep_learning.tensor import Tensor

    input_a, input_b = ensure_tensor(input_a), ensure_tensor(input_b)
    out = Tensor(input_a.data + input_b.data, requires_grad=input_a.requires_grad or input_b.requires_grad, dtype=input_a.data.dtype)
    out.depends_on = [input_a, input_b]

    def _backward() -> None:
        if input_a.requires_grad:
            grad_a = unbroadcast(out.grad, input_a.data.shape)
            input_a.grad = input_a.grad + grad_a if input_a.grad is not None else grad_a
            
        if input_b.requires_grad:
            grad_b = unbroadcast(out.grad, input_b.data.shape)
            input_b.grad = input_b.grad + grad_b if input_b.grad is not None else grad_b

    out._grad_func = _backward
    return out


def sub(input_a, input_b):
    """Subtract two tensors element-wise."""
    from deep_learning.tensor import Tensor

    input_a, input_b = ensure_tensor(input_a), ensure_tensor(input_b)
    out = Tensor(input_a.data - input_b.data, requires_grad=input_a.requires_grad or input_b.requires_grad, dtype=input_a.data.dtype)
    out.depends_on = [input_a, input_b]

    def _backward() -> None:
        if input_a.requires_grad:
            grad_a = unbroadcast(out.grad, input_a.data.shape)
            input_a.grad = input_a.grad + grad_a if input_a.grad is not None else grad_a
        if input_b.requires_grad:
            grad_b = unbroadcast(-out.grad, input_b.data.shape)
            input_b.grad = input_b.grad + grad_b if input_b.grad is not None else grad_b

    out._grad_func = _backward
    return out


def neg(input):
    """Negate the tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(-input.data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = -out.grad
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def mul(input_a, input_b):
    """Multiply two tensors element-wise."""
    from deep_learning.tensor import Tensor

    input_a, input_b = ensure_tensor(input_a), ensure_tensor(input_b)
    out = Tensor(input_a.data * input_b.data, requires_grad=input_a.requires_grad or input_b.requires_grad, dtype=input_a.data.dtype)
    out.depends_on = [input_a, input_b]

    def _backward() -> None:
        if input_a.requires_grad:
            grad_a = unbroadcast(input_b.data * out.grad, input_a.data.shape)
            input_a.grad = input_a.grad + grad_a if input_a.grad is not None else grad_a
        if input_b.requires_grad:
            grad_b = unbroadcast(input_a.data * out.grad, input_b.data.shape)
            input_b.grad = input_b.grad + grad_b if input_b.grad is not None else grad_b

    out._grad_func = _backward
    return out


def div(input_a, input_b):
    """Divide two tensors element-wise."""
    from deep_learning.tensor import Tensor

    input_a, input_b = ensure_tensor(input_a), ensure_tensor(input_b)
    out = Tensor(input_a.data / input_b.data, requires_grad=input_a.requires_grad or input_b.requires_grad, dtype=input_a.data.dtype)
    out.depends_on = [input_a, input_b]

    def _backward() -> None:
        if input_a.requires_grad:
            grad_a = unbroadcast(out.grad / input_b.data, input_a.data.shape)
            input_a.grad = input_a.grad + grad_a if input_a.grad is not None else grad_a
        if input_b.requires_grad:
            grad_b = unbroadcast(-input_a.data / (input_b.data ** 2) * out.grad, input_b.data.shape)
            input_b.grad = input_b.grad + grad_b if input_b.grad is not None else grad_b

    out._grad_func = _backward
    return out


def matmul(input_a, input_b):
    """Matrix multiplication of two tensors."""
    from deep_learning.tensor import Tensor

    input_a, input_b = ensure_tensor(input_a), ensure_tensor(input_b)
    out = Tensor(input_a.data @ input_b.data, requires_grad=input_a.requires_grad or input_b.requires_grad, dtype=input_a.data.dtype)
    out.depends_on = [input_a, input_b]

    def _backward() -> None:
        if input_a.requires_grad:
            grad = out.grad @ input_b.data.T
            input_a.grad = input_a.grad + grad if input_a.grad is not None else grad
        if input_b.requires_grad:
            grad = input_a.data.T @ out.grad
            input_b.grad = input_b.grad + grad if input_b.grad is not None else grad

    out._grad_func = _backward
    return out


def pow(input, power: Number):
    """Raise the tensor to a power."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(input.data ** power, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = power * (input.data ** (power - 1)) * out.grad
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


# Mathematical functions
def exp(input):
    """Compute the exponential of the tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out_data = np.exp(input.data)
    out = Tensor(out_data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = out_data * out.grad
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def log(input):
    """Compute the natural logarithm of the tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(np.log(input.data), requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = out.grad / input.data
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


# Activation functions
def linear(input, weight, bias=None):
    """Apply a linear transformation to the input tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    weight = ensure_tensor(weight)
    
    if bias is not None:
        bias = ensure_tensor(bias)
        out_data = input.data @ weight.data.T + bias.data
    else:
        out_data = input.data @ weight.data.T
    
    out = Tensor(out_data, requires_grad=input.requires_grad or weight.requires_grad or (bias.requires_grad if bias is not None else False))
    out.depends_on = [input, weight]
    
    if bias is not None:
        out.depends_on.append(bias)

    def _backward() -> None:
        if input.requires_grad:
            grad_input = out.grad @ weight.data
            input.grad = input.grad + grad_input if input.grad is not None else grad_input
        
        if weight.requires_grad:
            grad_weight = out.grad.T @ input.data
            weight.grad = weight.grad + grad_weight if weight.grad is not None else grad_weight
        
        if bias is not None and bias.requires_grad:
            grad_bias = np.sum(out.grad, axis=0)
            bias.grad = bias.grad + grad_bias if bias.grad is not None else grad_bias

    out._grad_func = _backward
    return out


def relu(input):
    """Apply the ReLU activation function."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(np.maximum(0, input.data), requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = np.where(input.data > 0, out.grad, 0)
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def tanh(input):
    """Apply the Tanh activation function."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    tanh_data = np.tanh(input.data)
    out = Tensor(tanh_data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = (1 - tanh_data ** 2) * out.grad
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def sigmoid(input):
    """Apply the Sigmoid activation function."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    sig = 1 / (1 + np.exp(-input.data))
    out = Tensor(sig, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad and out.grad is not None:
            grad = sig * (1 - sig) * out.grad
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def softmax(input, axis: int = -1):
    """Apply the Softmax activation function along a specified axis."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    exp_a = np.exp(input.data - np.max(input.data, axis=axis, keepdims=True))
    softmax_data = exp_a / np.sum(exp_a, axis=axis, keepdims=True)
    out = Tensor(softmax_data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = softmax_data * (out.grad - np.sum(out.grad * softmax_data, axis=axis, keepdims=True))
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


# Feature extraction functions
def conv2d(input, weight, bias=None):
    """
    Perform 2D convolution on input with stride=1, dilation=1, padding=0. 
    See: https://www.youtube.com/watch?v=Lakz2MoHy6o&t
    """
    from deep_learning.tensor import Tensor
    # TOOO: Add support for stride, dilation, and padding
    # TODO: This function takes too long to run, optimize it (risk is losing educational benefit)

    input = ensure_tensor(input)
    weight = ensure_tensor(weight)

    batch_size, in_channels, in_height, in_width = input.data.shape
    out_channels, weight_in_channels, kernel_height, kernel_width = weight.data.shape

    if bias is not None:
        if bias.data.shape != (out_channels,):
            raise ValueError(f"Bias shape {bias.data.shape} must match output channels {out_channels}")
        bias = ensure_tensor(bias)

    if weight_in_channels != in_channels:
        raise ValueError(f"Weight channels {weight_in_channels} must match input channels {in_channels}")
    
    # Compute output dimensions
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"Output dimensions ({out_height}, {out_width}) must be positive. Input size ({in_height}, {in_width}) too small for kernel ({kernel_height}, {kernel_width})")
    
    # Initialize output tensor
    out_data = np.zeros((batch_size, out_channels, out_height, out_width), dtype=input.data.dtype)
    for batch_index in range(batch_size):
        for out_channel in range(out_channels):
            for in_channel in range(in_channels):
                out_data[batch_index, out_channel] += correlate2d(input.data[batch_index, in_channel], weight.data[out_channel, in_channel])

    # Add bias if provided
    if bias is not None:
        out_data += bias.data.reshape(1, out_channels, 1, 1)

    out = Tensor(out_data, requires_grad=input.requires_grad or weight.requires_grad or (bias.requires_grad if bias is not None else False), dtype=input.data.dtype)
    out.depends_on = [input, weight]

    if bias is not None:
        out.depends_on.append(bias)

    def _backward() -> None:
        if input.requires_grad:
            grad_input = np.zeros_like(input.data)
            
            for batch_index in range(batch_size):
                for out_channel in range(out_channels):
                    for in_channel in range(in_channels):
                        padded_grad = np.pad(out.grad[batch_index, out_channel], ((kernel_height-1, kernel_height-1), (kernel_width-1, kernel_width-1)), mode='constant', constant_values=0)
                        grad_input[batch_index, in_channel] += convolve2d(padded_grad, weight.data[out_channel, in_channel])

            input.grad = input.grad + grad_input if input.grad is not None else grad_input
        
        if weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)
            
            for batch_index in range(batch_size):
                for out_channel in range(out_channels):
                    for in_channel in range(in_channels):
                        grad_weight[out_channel, in_channel] += correlate2d(input.data[batch_index, in_channel], out.grad[batch_index, out_channel])

            weight.grad = weight.grad + grad_weight if weight.grad is not None else grad_weight
        
        if bias is not None and bias.requires_grad:
            bias.grad = bias.grad + np.sum(out.grad, axis=(0, 2, 3)) if bias.grad is not None else np.sum(out.grad, axis=(0, 2, 3))

    out._grad_func = _backward
    return out


def max_pool2d(input, kernel_size: Tuple[int, int], stride: Optional[Tuple[int, int]] = None):
    """Perform 2D max pooling on input."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    batch_size, in_channels, in_height, in_width = input.data.shape
    kernel_height, kernel_width = kernel_size

    if stride is None:
        stride = kernel_size  # Updated default
    stride_height, stride_width = stride

    # Compute output dimensions
    out_height = (in_height - kernel_height) // stride_height + 1
    out_width = (in_width - kernel_width) // stride_width + 1

    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"Output dimensions ({out_height}, {out_width}) must be positive. Input size ({in_height}, {in_width}) too small for kernel ({kernel_height}, {kernel_width}) and stride ({stride_height}, {stride_width})")

    # Initialize output tensor and index map
    out_data = np.zeros((batch_size, in_channels, out_height, out_width), dtype=input.data.dtype)
    max_indices = np.zeros_like(out_data, dtype=np.int32)

    for batch_index in range(batch_size):
        for channel in range(in_channels):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride_height
                    h_end = h_start + kernel_height
                    w_start = j * stride_width
                    w_end = w_start + kernel_width

                    window = input.data[batch_index, channel, h_start:h_end, w_start:w_end]
                    max_index = np.argmax(window)
                    out_data[batch_index, channel, i, j] = window.flat[max_index]
                    max_indices[batch_index, channel, i, j] = max_index

    out = Tensor(out_data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad_input = np.zeros_like(input.data)

            for batch_index in range(batch_size):
                for channel in range(in_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            h_start = i * stride_height
                            w_start = j * stride_width

                            max_index = max_indices[batch_index, channel, i, j]
                            max_h = h_start + (max_index // kernel_width)
                            max_w = w_start + (max_index % kernel_width)

                            grad_input[batch_index, channel, max_h, max_w] += out.grad[batch_index, channel, i, j]

            input.grad = input.grad + grad_input if input.grad is not None else grad_input

    out._grad_func = _backward
    return out


# Regularization functions
def dropout(input, p: float = 0.5, training: bool = True):
    """Apply dropout regularization to the input tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in [0, 1], got {p}")
    
    if not training or p == 0.0:
        # During inference or when p=0, return input unchanged
        return input
    
    if p == 1.0:
        # When p=1, zero out everything
        out_data = np.zeros_like(input.data)
        out = Tensor(out_data, requires_grad=input.requires_grad)
        out.depends_on = [input]
        
        def _backward() -> None:
            if input.requires_grad:
                # Gradient is zero everywhere since output is zero
                grad = np.zeros_like(input.data)
                input.grad = input.grad + grad if input.grad is not None else grad
        
        out._grad_func = _backward
        return out
    
    # Generate random mask
    keep_prob = 1.0 - p
    mask = np.random.binomial(1, keep_prob, size=input.data.shape) / keep_prob
    
    # Apply mask and scale by 1/keep_prob to maintain expected value
    out_data = input.data * mask
    out = Tensor(out_data, requires_grad=input.requires_grad)
    out.depends_on = [input]
    
    def _backward() -> None:
        if input.requires_grad:
            # Gradient backpropagates only where mask is non-zero
            grad = out.grad * mask
            input.grad = input.grad + grad if input.grad is not None else grad
    
    out._grad_func = _backward
    return out


def batch_norm2d(input, running_mean, running_var, weight = None, bias = None, training: bool = True, momentum: float = 0.1, eps: float = EPSILON):
    """Applies 2D batch normalization on input of shape (N, C, H, W)."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    N, C, H, W = input.data.shape

    if weight is not None:
        weight = ensure_tensor(weight)
        if weight.data.shape != (C,):
            raise ValueError(f"Weight shape {weight.data.shape} must match number of channels {C}")
        
    if bias is not None:
        bias = ensure_tensor(bias)
        if bias.data.shape != (C,):
            raise ValueError(f"Bias shape {bias.data.shape} must match number of channels {C}")

    reduce_axes = (0, 2, 3)

    if training:
        mean = np.mean(input.data, axis=reduce_axes, keepdims=True)
        var = np.var(input.data, axis=reduce_axes, keepdims=True)

        # Update running statistics with Tensor objects
        running_mean.data = momentum * mean.squeeze() + (1 - momentum) * running_mean.data
        running_var.data = momentum * var.squeeze() + (1 - momentum) * running_var.data
    else:
        mean = running_mean.data.reshape(1, C, 1, 1)
        var = running_var.data.reshape(1, C, 1, 1)

    std = np.sqrt(var + eps)
    x_hat = (input.data - mean) / std

    out_data = x_hat
    if weight is not None:
        out_data = out_data * weight.data.reshape(1, C, 1, 1)
    if bias is not None:
        out_data = out_data + bias.data.reshape(1, C, 1, 1)

    requires_grad = input.requires_grad or (weight.requires_grad if weight else False) or (bias.requires_grad if bias else False)
    out = Tensor(out_data, requires_grad=requires_grad)
    out.depends_on = [input]
    if weight is not None:
        out.depends_on.append(weight)
    if bias is not None:
        out.depends_on.append(bias)

    def _backward():
        dx_hat = out.grad
        if weight is not None:
            dx_hat = dx_hat * weight.data.reshape(1, C, 1, 1)

        x_mu = input.data - mean
        inv_std = 1.0 / std
        N_HW = N * H * W

        dvar = np.sum(dx_hat * x_mu * -0.5 * inv_std**3, axis=reduce_axes, keepdims=True)
        dmean = np.sum(dx_hat * -inv_std, axis=reduce_axes, keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=reduce_axes, keepdims=True)

        grad_input = dx_hat * inv_std + dvar * 2 * x_mu / N_HW + dmean / N_HW

        if input.grad is None:
            input.grad = grad_input
        else:
            input.grad += grad_input

        if weight is not None and weight.requires_grad:
            grad_weight = np.sum(out.grad * x_hat, axis=(0, 2, 3))
            weight.grad = grad_weight if weight.grad is None else weight.grad + grad_weight

        if bias is not None and bias.requires_grad:
            grad_bias = np.sum(out.grad, axis=(0, 2, 3))
            bias.grad = grad_bias if bias.grad is None else bias.grad + grad_bias

    out._grad_func = _backward
    return out


# Reduction operations
def sum(input, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    """Compute the sum of the tensor along specified axes."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(input.data.sum(axis=axis, keepdims=keepdims), requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = out.grad
            a_grad = np.ones_like(input.data) * grad
            input.grad = input.grad + a_grad if input.grad is not None else a_grad

    out._grad_func = _backward
    return out


def mean(input, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    """Compute the mean of the tensor along specified axes."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(input.data.mean(axis=axis, keepdims=keepdims), requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = out.grad
            div = np.prod(input.data.shape if axis is None else np.array(input.data.shape)[axis])
            a_grad = np.ones_like(input.data) * grad / div
            input.grad = input.grad + a_grad if input.grad is not None else a_grad

    out._grad_func = _backward
    return out


# Shape manipulation
def flatten(input, start_dim: int = 0, end_dim: int = -1):
    """Flatten the tensor from start_dim to end_dim."""
    from deep_learning.tensor import Tensor
    input = ensure_tensor(input)

    ndim = input.data.ndim
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim

    if not (0 <= start_dim <= end_dim < ndim):
        raise ValueError(
            f"Invalid dimensions for flattening: start_dim={start_dim}, end_dim={end_dim}, "
            f"for tensor with {ndim} dimensions"
        )

    # Shape before start_dim, flattened part, shape after end_dim
    leading_shape = input.data.shape[:start_dim]
    flattened_dim = np.prod(input.data.shape[start_dim:end_dim + 1])
    trailing_shape = input.data.shape[end_dim + 1:]

    new_shape = leading_shape + (flattened_dim,) + trailing_shape
    out_data = input.data.reshape(new_shape)

    out = Tensor(out_data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad_input = out.grad.reshape(input.data.shape)
            input.grad = input.grad + grad_input if input.grad is not None else grad_input

    out._grad_func = _backward
    return out


def transpose(input):
    """Transpose the tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    out = Tensor(input.data.T, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            grad = out.grad.T
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def clip(input, min_val: Number, max_val: Number):
    """Clip the tensor values to a specified range."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)

    clipped_data = np.clip(input.data, min_val, max_val)
    out = Tensor(clipped_data, requires_grad=input.requires_grad, dtype=input.data.dtype)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad:
            # Gradient is passed through only where original data was not clipped
            mask = (input.data >= min_val) & (input.data <= max_val)
            grad = mask.astype(input.data.dtype) * out.grad
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def squeeze(input, axis: int = -1):
    """Remove dimensions of size 1 from the tensor."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    squeezed_data = input.data.squeeze(axis)
    # Ensure we don't squeeze away all dimensions for scalars
    if squeezed_data.ndim == 0 and input.data.ndim > 0:
        squeezed_data = squeezed_data.reshape(1)
    
    out = Tensor(squeezed_data, requires_grad=input.requires_grad)
    out.depends_on = [input]

    def _backward() -> None:
        if input.requires_grad and out.grad is not None:
            # Expand gradient back to original shape
            grad = np.expand_dims(out.grad, axis)
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


# Loss functions
def binary_cross_entropy(input, targets, reduction: str = "mean"):
    """Compute the binary cross-entropy loss with respect to the targets."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    targets = ensure_tensor(targets)

    if reduction != "mean":
        raise ValueError(f"Unsupported reduction method: {reduction}. Only 'mean' is supported for binary cross-entropy loss.")

    # Flatten both to 1D to avoid shape mismatches
    preds_flat = input.data.flatten()
    targets_flat = targets.data.flatten()
    
    clipped = np.clip(preds_flat, EPSILON, 1 - EPSILON)
    loss = -np.mean(targets_flat * np.log(clipped) + (1 - targets_flat) * np.log(1 - clipped))
    out = Tensor(loss, requires_grad=input.requires_grad)
    out.depends_on = [input, targets]

    def _backward() -> None:
        if input.requires_grad:
            grad = (clipped - targets_flat) / (clipped * (1 - clipped) * targets_flat.size)
            
            # Reshape gradient back to original preds shape
            grad = grad.reshape(input.data.shape)
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def cross_entropy(input, targets, reduction: str = "mean"):
    """Compute the cross-entropy loss with respect to the targets."""
    from deep_learning.tensor import Tensor
    
    input = ensure_tensor(input)
    targets = ensure_tensor(targets)

    if reduction != "mean":
        raise ValueError(f"Unsupported reduction method: {reduction}. Only 'mean' is supported for cross-entropy loss.")

    clipped = np.clip(input.data, EPSILON, 1 - EPSILON)

    loss = -np.mean(np.log(clipped[np.arange(len(targets.data)), targets.data.astype(int)]))

    out = Tensor(loss, requires_grad=input.requires_grad)
    out.depends_on = [input, targets]

    def _backward() -> None:
        if input.requires_grad:
            grad = np.zeros_like(input.data)
            # Compute the gradient w.r.t. the predicted probabilities
            grad[np.arange(len(targets.data)), targets.data.astype(int)] = -1 / clipped[np.arange(len(targets.data)), targets.data.astype(int)]
            grad = grad / len(targets.data)  # Mean over batch
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def mse_loss(input, targets, reduction: str = "mean"):
    """Compute the Mean Squared Error loss with respect to the targets."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    targets = ensure_tensor(targets)

    if reduction != "mean":
        raise ValueError(f"Unsupported reduction method: {reduction}. Only 'mean' is supported for MSE loss.")

    # Flatten both to 1D to avoid shape mismatches
    preds_flat = input.data.flatten()
    targets_flat = targets.data.flatten()

    loss = np.mean((preds_flat - targets_flat) ** 2)
    out = Tensor(loss, requires_grad=input.requires_grad)
    out.depends_on = [input, targets]

    def _backward() -> None:
        if input.requires_grad:
            grad = 2 * (preds_flat - targets_flat) / targets_flat.size
            
            # Reshape gradient back to original preds shape
            grad = grad.reshape(input.data.shape)
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out


def mae_loss(input, targets, reduction: str = "mean"):
    """Compute the Mean Absolute Error loss with respect to the targets."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    targets = ensure_tensor(targets)

    if reduction != "mean":
        raise ValueError(f"Unsupported reduction method: {reduction}. Only 'mean' is supported for MAE loss.")

    # Flatten both to 1D to avoid shape mismatches
    preds_flat = input.data.flatten()
    targets_flat = targets.data.flatten()

    loss = np.mean(np.abs(preds_flat - targets_flat))
    out = Tensor(loss, requires_grad=input.requires_grad)
    out.depends_on = [input, targets]

    def _backward() -> None:
        if input.requires_grad:
            grad = np.sign(preds_flat - targets_flat) / targets_flat.size
            
            # Reshape gradient back to original preds shape
            grad = grad.reshape(input.data.shape)
            input.grad = input.grad + grad if input.grad is not None else grad

    out._grad_func = _backward
    return out
