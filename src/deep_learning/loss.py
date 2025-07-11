from deep_learning.tensor import Tensor
from deep_learning import functional as F


class Loss:
    def __init__(self, reduction: str = "mean"):
        """Base class for loss functions. Only supports mean reduction by default."""
        self.reduction = reduction  # Default reduction method

    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the forward method to compute the loss."""
        return self.forward(*args, **kwargs)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between input and target tensors."""
        raise NotImplementedError("Loss function must implement the forward method.")


class MSELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        """Mean Squared Error Loss."""
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the mean squared error loss."""
        return F.mse_loss(input, target, reduction=self.reduction)
    

class MAELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        """Mean Absolute Error Loss."""
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the mean absolute error loss."""
        return F.mae_loss(input, target, reduction=self.reduction)


class BCELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        """Binary Cross Entropy Loss."""
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the binary cross entropy loss."""
        return F.binary_cross_entropy(input, target, self.reduction)
    

class CrossEntropyLoss(Loss):
    def __init__(self, reduction: str = "mean"):
        """Cross Entropy Loss for multi-class classification."""
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the cross entropy loss."""
        return F.cross_entropy(input, target, self.reduction)
