from deep_learning.tensor import Tensor
from deep_learning import functional as F


class Loss:
    def __init__(self, reduction: str):
        """Base class for loss functions. Only supports mean reduction by default."""
        self.reduction = "mean"  # Default reduction method

    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the forward method to compute the loss."""
        return self.forward(*args, **kwargs)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between input and target tensors."""
        raise NotImplementedError("Loss function must implement the forward method.")

# TODO: Implement Mean Squared Error (MSE) Loss and Mean Absolute Error (MAE) Loss


class BCELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        """Binary Cross Entropy Loss."""
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the binary cross entropy loss."""
        return F.binary_cross_entropy(input, target)
    

class CrossEntropyLoss(Loss):
    def __init__(self, reduction: str = "mean"):
        """Cross Entropy Loss for multi-class classification."""
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the cross entropy loss."""
        return F.cross_entropy(input, target)
