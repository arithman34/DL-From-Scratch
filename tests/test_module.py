import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from deep_learning.tensor import *
from deep_learning.module import *
from deep_learning.dataset import Dataset, DataLoader
from deep_learning.optimizers import SGD
from data.data_generator import get_classification_data


class BinaryClassifier(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.net = Sequential(
            Linear(in_features, hidden_features),
            ReLU(),
            Linear(hidden_features, out_features),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TestBinaryClassifier(unittest.TestCase):
    def setUp(self):
        # Config
        np.random.seed(42)
        torch.manual_seed(42)
        self.num_samples = 1000
        self.num_features = 5
        self.batch_size = 64
        self.num_epochs = 1000

        # Data
        X_train, X_test, y_train, y_test = get_classification_data(
            num_samples=self.num_samples,
            num_features=self.num_features,
            n_informative=3,
            n_redundant=1,
            n_repeated=1,
            num_classes=2,
            random_state=42
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # Custom Datasets
        self.train_dataset = Dataset(X_train, y_train)
        self.test_dataset = Dataset(X_test, y_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Custom Model
        self.model = BinaryClassifier(in_features=self.num_features, hidden_features=64, out_features=1)
        self.optimizer = SGD(self.model.parameters, lr=0.01)
        self.criterion = BCELoss()

    def test_initialization(self):
        self.assertIsInstance(self.model, BinaryClassifier)
        self.assertEqual(len(self.model.parameters), 4)  # 2 Linear layers: W and b each
        sample_input = self.train_dataset[0][0]
        output = self.model(sample_input)
        self.assertEqual(output.shape, (1,))

    def test_fit(self):
        initial_loss = None
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for x_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.data * len(x_batch.data)
                preds = (outputs.data > 0.5).astype(int).squeeze(-1)
                labels = y_batch.data.astype(int)
                total_correct += np.sum(preds == labels)
                total_samples += len(y_batch.data)

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            if epoch == 0:
                initial_loss = avg_loss

        self.avg_loss = avg_loss
        self.initial_loss = initial_loss
        self.custom_train_acc = accuracy

        self.assertLess(avg_loss, initial_loss, "Custom model did not learn (loss didn't decrease).")
        self.assertGreater(self.custom_train_acc, 0.80, "Training accuracy too low.")

    def test_predict(self):
        # Custom framework fit
        for _ in range(self.num_epochs):
            for x_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

        # Custom framework prediction
        test_correct = 0
        test_total = 0
        for x_batch, y_batch in self.test_loader:
            outputs = self.model(x_batch)
            preds = (outputs.data > 0.5).astype(int).squeeze(-1)
            labels = y_batch.data.astype(int)
            test_correct += np.sum(preds == labels)
            test_total += len(y_batch.data)

        custom_test_acc = test_correct / test_total

        # PyTorch setup
        class TorchNet(nn.Module):
            def __init__(self, in_features):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x)

        X_train_torch = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_torch = torch.tensor(self.y_train, dtype=torch.float32)
        X_test_torch = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(self.y_test, dtype=torch.float32)

        torch_train_dataset = data.TensorDataset(X_train_torch, y_train_torch)
        torch_test_dataset = data.TensorDataset(X_test_torch, y_test_torch)

        torch_train_loader = data.DataLoader(torch_train_dataset, batch_size=self.batch_size, shuffle=True)
        torch_test_loader = data.DataLoader(torch_test_dataset, batch_size=self.batch_size, shuffle=False)

        torch_model = TorchNet(self.num_features)
        torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
        torch_criterion = nn.BCELoss()

        for _ in range(self.num_epochs):
            torch_model.train()
            for X_batch, y_batch in torch_train_loader:
                torch_optimizer.zero_grad()
                outputs = torch_model(X_batch).squeeze(-1)
                loss = torch_criterion(outputs, y_batch)
                loss.backward()
                torch_optimizer.step()

        # Evaluate PyTorch
        torch_model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for X_batch, y_batch in torch_test_loader:
                model_output = torch_model(X_batch)
                preds = (model_output > 0.5).int().squeeze(-1)
                labels = y_batch.int()
                total_correct += (preds == labels).sum().item()
                total_samples += y_batch.size(0)
            torch_acc = total_correct / total_samples

        self.assertGreater(custom_test_acc, 0.80, "Custom test accuracy too low.")
        self.assertAlmostEqual(custom_test_acc, torch_acc, delta=0.05, msg="Accuracy differs too much from PyTorch.")
