import numpy as np
from typing import Any

from deep_learning.tensor import Tensor
from deep_learning import functional as F


class Module:
    def __init__(self):
        """Initialize the Module."""
        self._parameters = []
        self._modules = {}

    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the forward method of the module."""
        return self.forward(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the module."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Zero out the gradients of all parameters."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None

    def __setattr__(self, key: Any, value: Any) -> None:
        """Set an attribute, handling parameters and modules."""
        if isinstance(value, Module):
            self._modules[key] = value
            
        if isinstance(value, Tensor) and getattr(value, "requires_grad", False):
            self._parameters.append(value)
        super().__setattr__(key, value)

    @property
    def parameters(self) -> list[Tensor]:
        """Return all parameters of the module and its submodules."""
        params = self._parameters.copy()
        for module in self._modules.values():
            params += module._parameters
        return params
    

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        """Initialize a Sequential module with a list of modules."""
        super().__init__()
        self._modules = {f"module_{i}": module for i, module in enumerate(args)}
        self._parameters = [param for module in args for param in module._parameters]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the sequential modules."""
        for module in self._modules.values():
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize a Linear layer with weights and bias."""
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True, dtype=np.float64)  # He initialization
        self.bias = Tensor(np.zeros((out_features,)), requires_grad=True, dtype=np.float64)

        self._parameters = [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear layer."""
        return F.linear(x, self.weight, self.bias)
    

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        """Initialize a Dropout layer with a dropout probability."""
        super().__init__()
        self.p = p

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Forward pass through the dropout layer."""
        return F.dropout(x, self.p, training)


# FIXME: Freeze gradient tracking during inference
# TODO: Add reshape layer
# TODO: Add batch normalization layer
# TODO: Add conv1d, conv2d layers


class ReLU(Module):
    def __init__(self) -> None:
        """Initialize a ReLU activation module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ReLU activation."""
        return F.relu(x)


class Tanh(Module):
    def __init__(self) -> None:
        """Initialize a Tanh activation module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Tanh activation."""
        return F.tanh(x)


class Sigmoid(Module):
    def __init__(self) -> None:
        """Initialize a Sigmoid activation module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Sigmoid activation."""
        return F.sigmoid(x)
    

class Softmax(Module):
    def __init__(self) -> None:
        """Initialize a Softmax activation module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Softmax activation."""
        return F.softmax(x, axis=-1)
