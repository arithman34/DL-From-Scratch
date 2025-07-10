import numpy as np
import time

from deep_learning.dataset import Dataset, DataLoader
from deep_learning.module import Module, Linear, ReLU, Sigmoid, Sequential
from deep_learning.optimizers import Adam
from deep_learning.loss import *
from deep_learning.tensor import Tensor
from data.data_generator import get_classification_data


class BinaryClassifier(Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = Sequential(
            Linear(in_features, hidden_features),
            ReLU(),
            Linear(hidden_features, out_features),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_binary_classifier():
    # Config
    np.random.seed(42)
    num_samples = 10000
    num_features = 5
    batch_size = 128
    num_epochs = 100

    # Data
    X_train, X_test, y_train, y_test = get_classification_data(
        num_samples=num_samples,
        num_features=num_features,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        num_classes=2,
        random_state=42
    )

    # Datasets
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = BinaryClassifier(in_features=num_features, hidden_features=64, out_features=1)
    
    # Optimizer
    optimizer = Adam(model.parameters, lr=0.001)

    # Criterion
    criterion = BCELoss()

    train_losses = []
    train_accuracies = []

    model.train()  # Set to training mode
    print("Starting training...")
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.data * len(x_batch.data)
            preds = (outputs.data > 0.5).astype(int).squeeze()
            labels = y_batch.data.astype(int)
            total_correct += np.sum(preds == labels)
            total_samples += len(y_batch.data)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Evaluation
    print("\nEvaluating on test set...")
    total_correct = 0
    total_samples = 0
    model.eval()  # Set to evaluation mode
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        preds = (outputs.data > 0.5).astype(int).squeeze()
        labels = y_batch.data.astype(int)
        total_correct += np.sum(preds == labels)
        total_samples += len(y_batch.data)

    test_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Training and evaluation completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    train_binary_classifier()
