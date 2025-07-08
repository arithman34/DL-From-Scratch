import numpy as np
from typing import Optional, Union, Tuple
import numpy.typing as npt

from deep_learning.backend import EPSILON

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
        out_data = input.data @ weight.data + bias.data
    else:
        out_data = input.data @ weight.data
    
    out = Tensor(out_data, requires_grad=input.requires_grad or weight.requires_grad or (bias.requires_grad if bias is not None else False))
    out.depends_on = [input, weight]
    
    if bias is not None:
        out.depends_on.append(bias)

    def _backward() -> None:
        if input.requires_grad:
            grad_input = out.grad @ weight.data.T
            input.grad = input.grad + grad_input if input.grad is not None else grad_input
        
        if weight.requires_grad:
            grad_weight = input.data.T @ out.grad
            weight.grad = weight.grad + grad_weight if weight.grad is not None else grad_weight
        
        if bias is not None and bias.requires_grad:
            bias.grad = bias.grad + np.sum(out.grad, axis=0) if bias.grad is not None else np.sum(out.grad, axis=0)

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
def binary_cross_entropy(input, targets):
    """Compute the binary cross-entropy loss with respect to the targets."""
    from deep_learning.tensor import Tensor

    input = ensure_tensor(input)
    targets = ensure_tensor(targets)

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


def cross_entropy(input, targets):
    """Compute the cross-entropy loss with respect to the targets."""
    from deep_learning.tensor import Tensor
    
    input = ensure_tensor(input)
    targets = ensure_tensor(targets)

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
