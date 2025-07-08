import unittest
import numpy as np

from deep_learning.module import *


class TestModule(unittest.TestCase):
    def test_module_initialization(self):
        module = Module()
        self.assertIsInstance(module, Module)

    def test_linear_initialization(self):
        linear = Linear(5, 10)
        self.assertEqual(linear.weight.shape, (5, 10))
        self.assertEqual(linear.bias.shape, (10,))
        self.assertTrue(linear.weight.requires_grad)
        self.assertTrue(linear.bias.requires_grad)

    def test_linear_forward(self):
        linear = Linear(5, 10)
        input_tensor = Tensor(np.random.randn(64, 5), requires_grad=True)
        output_tensor = linear(input_tensor)
        self.assertEqual(output_tensor.shape, (64, 10))

    def test_relu_forward(self):
        relu = ReLU()
        input_tensor = Tensor(np.random.randn(64, 10), requires_grad=True)
        output_tensor = relu(input_tensor)
        self.assertEqual(output_tensor.shape, (64, 10))
        self.assertTrue(np.all(output_tensor.data >= 0))

    def test_sigmoid_forward(self):
        sigmoid = Sigmoid()
        input_tensor = Tensor(np.random.randn(64, 10), requires_grad=True)
        output_tensor = sigmoid(input_tensor)
        self.assertEqual(output_tensor.shape, (64, 10))
        self.assertTrue(np.all((output_tensor.data >= 0) & (output_tensor.data <= 1)))

    def test_softmax_forward(self):
        softmax = Softmax()
        input_tensor = Tensor(np.random.randn(64, 10), requires_grad=True)
        output_tensor = softmax(input_tensor)
        self.assertEqual(output_tensor.shape, (64, 10))
        self.assertTrue(np.allclose(output_tensor.data.sum(axis=-1), 1.0))

    def test_sequential_forward(self):
        model = Sequential(Linear(5, 10), ReLU(), Linear(10, 1), Sigmoid())
        input_tensor = Tensor(np.random.randn(64, 5), requires_grad=True)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (64, 1))
        self.assertTrue(np.all((output_tensor.data >= 0) & (output_tensor.data <= 1)))
