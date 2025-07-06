import numpy as np

from deep_learning.tensor import Tensor


class Module:
    def __init__(self):
        self._parameters = []
        self._modules = {}

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            self._modules[key] = value
            
        if isinstance(value, Tensor) and getattr(value, "requires_grad", False):
            self._parameters.append(value)
        super().__setattr__(key, value)

    @property
    def parameters(self):
        params = self._parameters.copy()
        for module in self._modules.values():
            params += module._parameters
        return params
    

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self._modules = {f"module_{i}": module for i, module in enumerate(args)}
        self._parameters = [param for module in args for param in module._parameters]

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weights = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True, dtype=np.float64)  # He initialization
        self.bias = Tensor(np.zeros((out_features,)), requires_grad=True, dtype=np.float64)
        self._parameters = [self.weights, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights + self.bias


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    

class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    

class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=-1)
