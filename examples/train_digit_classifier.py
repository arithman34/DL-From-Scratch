import time
import numpy as np

from deep_learning.dataset import Dataset, DataLoader
from deep_learning.module import Module, Linear, ReLU, Sequential, Conv2d, Softmax, MaxPool2d
from deep_learning.optimizers import Adam
from deep_learning.loss import CrossEntropyLoss
from deep_learning.tensor import Tensor
from data.data_generator import get_mnist_data


class MNISTClassifier(Module):
    def __init__(self, in_channels, hidden_features, out_features):
        super().__init__()

        # Feature extractor
        self.feature_extractor = Sequential(
            Conv2d(in_channels, hidden_features, kernel_size=(3, 3)),  # -> (hidden_features, 26, 26)
            ReLU(),
            MaxPool2d(kernel_size=(2, 2)),  # -> (hidden_features, 13, 13)
            Conv2d(hidden_features, hidden_features * 2, kernel_size=(3, 3)),  # -> (hidden_features*2, 11, 11)
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))  # -> (hidden_features*2, 5, 5)
        )

        final_channels = hidden_features * 2
        flattened_size = final_channels * 5 * 5

        # Fully connected layers
        self.fc = Sequential(
            Linear(flattened_size, hidden_features),
            ReLU(),
            Linear(hidden_features, out_features),
            Softmax()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        batch_size = x.shape[0]
        x = Tensor(x.data.reshape(batch_size, -1), requires_grad=x.requires_grad)
        return self.fc(x)


def train_digit_classifier():
    """Train a neural network to classify on the MNIST dataset."""
    # Config
    np.random.seed(42)
    num_samples = 64  # Use smaller dataset for faster training (still takes a while)
    batch_size = 32 
    num_epochs = 10

    # Data
    X_train, X_test, y_train, y_test = get_mnist_data(num_samples=num_samples, random_state=42)

    # Datasets
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    in_channels = 1  # MNIST images have 1 channel
    model = MNISTClassifier(in_channels=in_channels, hidden_features=32, out_features=10)
    
    # Optimizer
    optimizer = Adam(model.parameters, lr=0.001)

    # Criterion
    criterion = CrossEntropyLoss()

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
            preds = np.argmax(outputs.data, axis=1)
            labels_np = y_batch.data.astype(int)
            total_correct += np.sum(preds == labels_np)
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
        loss = criterion(outputs, y_batch)

        preds = np.argmax(outputs.data, axis=1)
        labels_np = y_batch.data.astype(int)
        total_correct += np.sum(preds == labels_np)
        total_samples += len(y_batch.data)

    test_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Training and evaluation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train_digit_classifier()
