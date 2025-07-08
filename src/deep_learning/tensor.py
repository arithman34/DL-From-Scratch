import numpy as np
from typing import Iterable, List, Optional, Union, Tuple, Any
import numpy.typing as npt

from deep_learning.backend import EPSILON


class Tensor:
    def __init__(self, data: Union[Iterable, npt.NDArray], requires_grad: bool = False, depends_on: Optional[List["Tensor"]] = None, dtype: np.dtype = np.float64) -> None:
        """Initialize a Tensor object with data and the option to track gradients with autodiff."""
        self.data = np.asarray(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad = None
        self._grad_func = None

    def __repr__(self) -> str:
        """String representation of the Tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: Optional[npt.NDArray] = None) -> None:
        """Compute the gradient of the tensor with respect to its dependencies."""
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad
        topo_order = []
        visited = set()

        def build_topo(tensor: "Tensor") -> None:
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor.depends_on:
                    build_topo(parent)
                topo_order.append(tensor)

        build_topo(self)

        for tensor in reversed(topo_order):
            if tensor.requires_grad and tensor._grad_func is not None:
                tensor._grad_func()

    def zero_grad(self) -> None:
        """Zero out the gradients of the tensor and its dependencies."""
        if self.requires_grad:
            self.grad = None
            for tensor in self.depends_on:
                tensor.zero_grad()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape
    
    @property
    def T(self) -> "Tensor":
        """Return the transpose of the tensor."""
        return Transpose()(self)
    
    # Operator overloading
    def __add__(self, other: "Tensor") -> "Tensor":
        """Add two tensors."""
        return Add()(self, other)
    
    def __iadd__(self, other: "Tensor") -> "Tensor":
        """In-place addition of two tensors."""
        return Add()(other, self)

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Subtract two tensors."""
        return Sub()(self, other)
    
    def __isub__(self, other: "Tensor") -> "Tensor":
        """In-place subtraction of two tensors."""
        return Sub()(other, self)

    def __neg__(self) -> "Tensor":
        """Negate the tensor."""
        return Neg()(self)

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Multiply two tensors."""
        return Mul()(self, other)
    
    def __imul__(self, other: "Tensor") -> "Tensor":
        """In-place multiplication of two tensors."""
        return Mul()(other, self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication of two tensors."""
        return MatMul()(self, other)
    
    def __imatmul__(self, other: "Tensor") -> "Tensor":
        """In-place matrix multiplication of two tensors."""
        return MatMul()(other, self)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        """Divide two tensors."""
        return Div()(self, other)
    
    def __itruediv__(self, other: "Tensor") -> "Tensor":
        """In-place division of two tensors."""
        return Div()(other, self)
    
    def __pow__(self, power: Union[int, float]) -> "Tensor":
        """Raise the tensor to a power."""
        return Pow()(self, power)
    
    def __ipow__(self, power: Union[int, float]) -> "Tensor":
        """In-place exponentiation of the tensor."""
        return Pow()(self, power)
    
    # Functional operations
    def exp(self) -> "Tensor":
        """Compute the exponential of the tensor."""
        return Exp()(self)

    def log(self) -> "Tensor":
        """Compute the natural logarithm of the tensor."""
        return Log()(self)

    def relu(self) -> "Tensor":
        """Apply the ReLU activation function."""
        return ReLU()(self)
    
    def tanh(self) -> "Tensor":
        """Apply the Tanh activation function."""
        return Tanh()(self)

    def sigmoid(self) -> "Tensor":
        """Apply the Sigmoid activation function."""
        return Sigmoid()(self)
    
    def softmax(self, axis: int = -1) -> "Tensor":
        """Apply the Softmax activation function along a specified axis."""
        return Softmax()(self, axis)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Compute the sum of the tensor along specified axes."""
        return Sum()(self, axis, keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Compute the mean of the tensor along specified axes."""
        return Mean()(self, axis, keepdims)
    
    def clip(self, min_val: Union[int, float], max_val: Union[int, float]) -> "Tensor":
        """Clip the tensor values to a specified range."""
        return Clip()(self, min_val, max_val)
    
    def squeeze(self, axis: int = -1) -> "Tensor":
        """Remove dimensions of size 1 from the tensor."""
        return Squeeze()(self, axis)

    def bce(self, targets: "Tensor") -> "Tensor":
        """Compute the binary cross-entropy loss with respect to the targets."""
        return BCELoss()(self, targets)
    
    def cross_entropy(self, targets: "Tensor") -> "Tensor":
        """Compute the cross-entropy loss with respect to the targets."""
        return CrossEntropyLoss()(self, targets)


def ensure_tensor(obj: Union["Tensor", Any]) -> "Tensor":
    """Ensure that the input is a Tensor object, converting if necessary."""
    return obj if isinstance(obj, Tensor) else Tensor(obj)


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


class Function:
    def __call__(self, *args: Any) -> Tensor:
        """Call the function with the provided arguments."""
        raise NotImplementedError()


class Add(Function):
    def __call__(self, a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
        """Add two tensors element-wise."""
        a, b = ensure_tensor(a), ensure_tensor(b)
        out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad, dtype=a.data.dtype)
        out.depends_on = [a, b]

        def _backward() -> None:
            if a.requires_grad:
                grad_a = unbroadcast(out.grad, a.data.shape)
                a.grad = a.grad + grad_a if a.grad is not None else grad_a
                
            if b.requires_grad:
                grad_b = unbroadcast(out.grad, b.data.shape)
                b.grad = b.grad + grad_b if b.grad is not None else grad_b

        out._grad_func = _backward
        return out


class Sub(Function):
    def __call__(self, a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
        """Subtract two tensors element-wise."""
        a, b = ensure_tensor(a), ensure_tensor(b)
        out = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad, dtype=a.data.dtype)
        out.depends_on = [a, b]

        def _backward() -> None:
            if a.requires_grad:
                grad_a = unbroadcast(out.grad, a.data.shape)
                a.grad = a.grad + grad_a if a.grad is not None else grad_a
            if b.requires_grad:
                grad_b = unbroadcast(-out.grad, b.data.shape)
                b.grad = b.grad + grad_b if b.grad is not None else grad_b

        out._grad_func = _backward
        return out


class Neg(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Negate the tensor."""
        a = ensure_tensor(a)
        out = Tensor(-a.data, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = -out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class Mul(Function):
    def __call__(self, a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
        """Multiply two tensors element-wise."""
        a, b = ensure_tensor(a), ensure_tensor(b)
        out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad, dtype=a.data.dtype)
        out.depends_on = [a, b]

        def _backward() -> None:
            if a.requires_grad:
                grad_a = unbroadcast(b.data * out.grad, a.data.shape)
                a.grad = a.grad + grad_a if a.grad is not None else grad_a
            if b.requires_grad:
                grad_b = unbroadcast(a.data * out.grad, b.data.shape)
                b.grad = b.grad + grad_b if b.grad is not None else grad_b

        out._grad_func = _backward
        return out


class Div(Function):
    def __call__(self, a: Union[Tensor, Any], b: Union[Tensor, Any]) -> Tensor:
        """Divide two tensors element-wise."""
        a, b = ensure_tensor(a), ensure_tensor(b)
        out = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad, dtype=a.data.dtype)
        out.depends_on = [a, b]

        def _backward() -> None:
            if a.requires_grad:
                grad_a = unbroadcast(out.grad / b.data, a.data.shape)
                a.grad = a.grad + grad_a if a.grad is not None else grad_a
            if b.requires_grad:
                grad_b = unbroadcast(-a.data / (b.data ** 2) * out.grad, b.data.shape)
                b.grad = b.grad + grad_b if b.grad is not None else grad_b

        out._grad_func = _backward
        return out


class MatMul(Function):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication of two tensors."""
        a, b = ensure_tensor(a), ensure_tensor(b)
        out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad, dtype=a.data.dtype)
        out.depends_on = [a, b]

        def _backward() -> None:
            if a.requires_grad:
                grad = out.grad @ b.data.T
                a.grad = a.grad + grad if a.grad is not None else grad
            if b.requires_grad:
                grad = a.data.T @ out.grad
                b.grad = b.grad + grad if b.grad is not None else grad

        out._grad_func = _backward
        return out


class Pow(Function):
    def __call__(self, a: Tensor, power: Union[int, float]) -> Tensor:
        """Raise the tensor to a power."""
        a = ensure_tensor(a)
        out = Tensor(a.data ** power, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = power * (a.data ** (power - 1)) * out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class Exp(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Compute the exponential of the tensor."""
        a = ensure_tensor(a)
        out_data = np.exp(a.data)
        out = Tensor(out_data, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = out_data * out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class Log(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Compute the natural logarithm of the tensor."""
        a = ensure_tensor(a)
        out = Tensor(np.log(a.data), requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = out.grad / a.data
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class ReLU(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Apply the ReLU activation function."""
        a = ensure_tensor(a)
        out = Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = np.where(a.data > 0, out.grad, 0) * out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out
    

class Tanh(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Apply the Tanh activation function."""
        a = ensure_tensor(a)
        tanh_data = np.tanh(a.data)
        out = Tensor(tanh_data, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = (1 - tanh_data ** 2) * out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class Sigmoid(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Apply the Sigmoid activation function."""
        a = ensure_tensor(a)
        sig = 1 / (1 + np.exp(-a.data))
        out = Tensor(sig, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad and out.grad is not None:
                grad = sig * (1 - sig) * out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out
    

class Softmax(Function):
    def __call__(self, a: Tensor, axis: int = -1) -> Tensor:
        """Apply the Softmax activation function along a specified axis."""
        a = ensure_tensor(a)
        exp_a = np.exp(a.data - np.max(a.data, axis=axis, keepdims=True))
        softmax_data = exp_a / np.sum(exp_a, axis=axis, keepdims=True)
        out = Tensor(softmax_data, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = softmax_data * (out.grad - np.sum(out.grad * softmax_data, axis=axis, keepdims=True))
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class Sum(Function):
    def __call__(self, a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """Compute the sum of the tensor along specified axes."""
        a = ensure_tensor(a)
        out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = out.grad
                a_grad = np.ones_like(a.data) * grad
                a.grad = a.grad + a_grad if a.grad is not None else a_grad

        out._grad_func = _backward
        return out


class Mean(Function):
    def __call__(self, a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """Compute the mean of the tensor along specified axes."""
        a = ensure_tensor(a)
        out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = out.grad
                div = np.prod(a.data.shape if axis is None else np.array(a.data.shape)[axis])
                a_grad = np.ones_like(a.data) * grad / div
                a.grad = a.grad + a_grad if a.grad is not None else a_grad

        out._grad_func = _backward
        return out


class Transpose(Function):
    def __call__(self, a: Tensor) -> Tensor:
        """Transpose the tensor."""
        a = ensure_tensor(a)
        out = Tensor(a.data.T, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                grad = out.grad.T
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out
    

class Clip(Function):
    def __call__(self, a: Tensor, min_val: Union[int, float], max_val: Union[int, float]) -> Tensor:
        """Clip the tensor values to a specified range."""
        a = ensure_tensor(a)

        clipped_data = np.clip(a.data, min_val, max_val)
        out = Tensor(clipped_data, requires_grad=a.requires_grad, dtype=a.data.dtype)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad:
                # Gradient is passed through only where original data was not clipped
                mask = (a.data >= min_val) & (a.data <= max_val)
                grad = mask.astype(a.data.dtype) * out.grad
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out


class BCELoss(Function):
    def __call__(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Compute the binary cross-entropy loss with respect to the targets."""
        preds = ensure_tensor(preds)
        targets = ensure_tensor(targets)

        # Flatten both to 1D to avoid shape mismatches
        preds_flat = preds.data.flatten()
        targets_flat = targets.data.flatten()
        
        clipped = np.clip(preds_flat, EPSILON, 1 - EPSILON)
        loss = -np.mean(targets_flat * np.log(clipped) + (1 - targets_flat) * np.log(1 - clipped))
        out = Tensor(loss, requires_grad=preds.requires_grad)
        out.depends_on = [preds, targets]

        def _backward() -> None:
            if preds.requires_grad:
                grad = (clipped - targets_flat) / (clipped * (1 - clipped) * targets_flat.size)
                
                # Reshape gradient back to original preds shape
                grad = grad.reshape(preds.data.shape)
                preds.grad = preds.grad + grad if preds.grad is not None else grad

        out._grad_func = _backward
        return out


class CrossEntropyLoss(Function):
    def __call__(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Compute the cross-entropy loss with respect to the targets."""
        preds = ensure_tensor(preds)
        targets = ensure_tensor(targets)

        clipped = np.clip(preds.data, EPSILON, 1 - EPSILON)

        loss = -np.mean(np.log(clipped[np.arange(len(targets.data)), targets.data.astype(int)]))

        out = Tensor(loss, requires_grad=preds.requires_grad)
        out.depends_on = [preds, targets]

        def _backward() -> None:
            if preds.requires_grad:
                grad = np.zeros_like(preds.data)
                # Compute the gradient w.r.t. the predicted probabilities
                grad[np.arange(len(targets.data)), targets.data.astype(int)] = -1 / clipped[np.arange(len(targets.data)), targets.data.astype(int)]
                grad = grad / len(targets.data)  # Mean over batch
                preds.grad = preds.grad + grad if preds.grad is not None else grad

        out._grad_func = _backward
        return out


class Squeeze(Function):
    def __call__(self, a: Tensor, axis: int = -1) -> Tensor:
        """Remove dimensions of size 1 from the tensor."""
        a = ensure_tensor(a)
        squeezed_data = a.data.squeeze(axis)
        # Ensure we don't squeeze away all dimensions for scalars
        if squeezed_data.ndim == 0 and a.data.ndim > 0:
            squeezed_data = squeezed_data.reshape(1)
        
        out = Tensor(squeezed_data, requires_grad=a.requires_grad)
        out.depends_on = [a]

        def _backward() -> None:
            if a.requires_grad and out.grad is not None:
                # Expand gradient back to original shape
                grad = np.expand_dims(out.grad, axis)
                a.grad = a.grad + grad if a.grad is not None else grad

        out._grad_func = _backward
        return out
