import numpy as np
from src.deep_learning.tensor import Tensor
from src.deep_learning.backend import EPSILON


class Optimizer:
    def __init__(self, parameters: list[Tensor], lr: float=0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None

    def step(self) -> None:
        raise NotImplementedError("Optimizer step() method must be overridden.")



class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float = 0.01) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * np.array(param.grad)


class Adam(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = EPSILON) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [Tensor(np.zeros_like(param.data), requires_grad=False) for param in parameters]
        self.v = [Tensor(np.zeros_like(param.data), requires_grad=False) for param in parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.m[i].data = self.beta1 * np.array(self.m[i].data) + (1 - self.beta1) * np.array(param.grad.data)
                self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (np.array(param.grad.data) ** 2)

                m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
                v_hat = self.v[i].data / (1 - self.beta2 ** self.t)

                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
