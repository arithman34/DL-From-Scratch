import numpy as np
from typing import Any

from deep_learning.tensor import Tensor
from deep_learning import functional as F


class Module:
    def __init__(self):
        """Initialize the Module."""
        self._parameters = []
        self._modules = {}
        self.training = True  # Default to training mode

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
    
    def train(self, mode: bool = True) -> "Module":
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        
        # Some modules may have parameters that were set to not require gradients so we need to maintain that state when switching modes.
        for param in self.parameters:
            param._original_requires_grad = getattr(param, '_original_requires_grad', param.requires_grad)
            if mode:
                param.requires_grad = param._original_requires_grad
            else:
                param.requires_grad = False
        
        return self

    def eval(self) -> "Module":
        """Set the module in evaluation mode."""
        return self.train(False)


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
        self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(2. / in_features), requires_grad=True, dtype=np.float64)  # He initialization
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the dropout layer."""
        return F.dropout(x, self.p, self.training)
    

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]) -> None:
        """Initialize a Conv2d layer with weights and bias."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights and bias
        self.weight = Tensor(np.random.randn(out_channels, in_channels, *kernel_size) * np.sqrt(2. / (in_channels * np.prod(kernel_size))), requires_grad=True, dtype=np.float64)  # He initialization
        self.bias = Tensor(np.zeros((out_channels,)), requires_grad=True, dtype=np.float64)

        self._parameters = [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Conv2d layer."""
        return F.conv2d(x, self.weight, self.bias)


class MaxPool2d(Module):
    def __init__(self, kernel_size: tuple[int, int], stride: tuple[int, int] = None) -> None:
        """Initialize a MaxPool2d layer."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MaxPool2d layer."""
        return F.max_pool2d(x, self.kernel_size, self.stride)


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
