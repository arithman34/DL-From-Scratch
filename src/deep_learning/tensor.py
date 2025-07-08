import numpy as np
from typing import Iterable, List, Optional, Union, Tuple
import numpy.typing as npt

Number = Union[int, float]


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
        from deep_learning import functional as F
        return F.transpose(self)
    
    def __add__(self, other: "Tensor") -> "Tensor":
        """Add another tensor to this tensor."""
        from deep_learning import functional as F
        return F.add(self, other)
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        """Subtract another tensor from this tensor."""
        from deep_learning import functional as F
        return F.sub(self, other)
    
    def __neg__(self) -> "Tensor":
        """Negate this tensor."""
        from deep_learning import functional as F
        return F.neg(self)
    
    def __mul__(self, other: "Tensor") -> "Tensor":
        """Multiply this tensor by another tensor element-wise."""
        from deep_learning import functional as F
        return F.mul(self, other)
    
    def __truediv__(self, other: "Tensor") -> "Tensor":
        """Divide this tensor by another tensor element-wise."""
        from deep_learning import functional as F
        return F.div(self, other)
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiply this tensor with another tensor."""
        from deep_learning import functional as F
        return F.matmul(self, other)
    
    def __pow__(self, power: Number) -> "Tensor":
        """Raise this tensor to a power."""
        from deep_learning import functional as F
        return F.pow(self, power)
