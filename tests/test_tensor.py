import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from deep_learning.tensor import Tensor
from deep_learning import functional as F


class TestTensor(unittest.TestCase):
    def test_tensor_initialization(self):
        # Test initialization of tensors
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=False)

        # Check that the data is correctly stored
        np.testing.assert_array_equal(x.data, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(y.data, np.array([3.0, 4.0]))

        # Check the requires_grad flag
        self.assertTrue(x.requires_grad)
        self.assertFalse(y.requires_grad)

        # Check that gradients are None initially
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_tensor_transpose(self):
        # Test the transpose operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = x.T
        torch_y = torch_x.T

        # Check the result of transpose
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_shape(self):
        # Test the shape property
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)

        # Check the shape of the tensor
        self.assertEqual(x.shape, (2, 2))


class TestAddition(unittest.TestCase):
    def test_tensor_addition(self):
        # Test the addition operation
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_y = torch.tensor(y.data, requires_grad=False)

        z = x + y
        torch_z = torch_x + torch_y

        # Check the result of the addition
        np.testing.assert_array_equal(z.data, torch_z.numpy())

    def test_tensor_addition_backward(self):
        # Test backward propagation of gradients
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_y = torch.tensor(y.data, requires_grad=True)

        z = x + y
        torch_z = torch_x + torch_y

        # Sum the output to reduce it to a scalar
        torch_z_sum = torch_z.sum()

        # Call backward() to calculate gradients
        z.backward()
        torch_z_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_equal(y.grad, torch_y.grad.numpy())


class TestSubtraction(unittest.TestCase):
    def test_tensor_subtraction(self):
        # Test the subtraction operation
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_y = torch.tensor(y.data, requires_grad=False)

        z = x - y
        torch_z = torch_x - torch_y

        # Check the result of subtraction
        np.testing.assert_array_equal(z.data, torch_z.numpy())

    def test_tensor_subtraction_backward(self):
        # Test backward propagation of gradients
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_y = torch.tensor(y.data, requires_grad=True)

        z = x - y
        torch_z = torch_x - torch_y

        # Sum the output to reduce it to a scalar
        torch_z_sum = torch_z.sum()

        # Call backward() to calculate gradients
        z.backward()
        torch_z_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_equal(y.grad, torch_y.grad.numpy())


class TestNegation(unittest.TestCase):
    def test_tensor_negation(self):
        # Test the negation operation
        x = Tensor(np.array([1.0, -2.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = -x
        torch_y = -torch_x

        # Check the result of negation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_negation_backward(self):
        # Test backward propagation of gradients
        x = Tensor(np.array([1.0, -2.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = -x
        torch_y = -torch_x

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestMultiplication(unittest.TestCase):
    def test_tensor_multiplication(self):
        # Test the multiplication operation
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_y = torch.tensor(y.data, requires_grad=False)

        z = x * y
        torch_z = torch_x * torch_y

        # Check the result of multiplication
        np.testing.assert_array_equal(z.data, torch_z.numpy())

    def test_tensor_multiplication_backward(self):
        # Test backward propagation of gradients
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_y = torch.tensor(y.data, requires_grad=True)

        z = x * y
        torch_z = torch_x * torch_y

        # Sum the output to reduce it to a scalar
        torch_z_sum = torch_z.sum()

        # Call backward() to calculate gradients
        z.backward()
        torch_z_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_equal(y.grad, torch_y.grad.numpy())


class TestMatrixMultiplication(unittest.TestCase):
    def test_tensor_matmul(self):
        # Test the matrix multiplication operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        y = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_y = torch.tensor(y.data, requires_grad=False)

        z = x @ y
        torch_z = torch_x @ torch_y

        # Check the result of matrix multiplication
        np.testing.assert_array_equal(z.data, torch_z.numpy())

    def test_tensor_matmul_backward(self):
        # Test backward propagation of gradients for matrix multiplication
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_y = torch.tensor(y.data, requires_grad=True)

        z = x @ y
        torch_z = torch_x @ torch_y

        # Sum the output to reduce it to a scalar
        torch_z_sum = torch_z.sum()

        # Call backward() to calculate gradients
        z.backward()
        torch_z_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_equal(y.grad, torch_y.grad.numpy())


class TestDivision(unittest.TestCase):
    def test_tensor_division(self):
        # Test the division operation
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_y = torch.tensor(y.data, requires_grad=False)

        z = x / y
        torch_z = torch_x / torch_y

        # Check the result of division
        np.testing.assert_array_equal(z.data, torch_z.numpy())

    def test_tensor_division_backward(self):
        # Test backward propagation of gradients
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_y = torch.tensor(y.data, requires_grad=True)

        z = x / y
        torch_z = torch_x / torch_y

        # Sum the output to reduce it to a scalar
        torch_z_sum = torch_z.sum()

        # Call backward() to calculate gradients
        z.backward()
        torch_z_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_equal(y.grad, torch_y.grad.numpy())


class TestPower(unittest.TestCase):
    def test_tensor_power(self):
        # Test the power operation
        x = Tensor(np.array([2.0, 3.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = x ** 2
        torch_y = torch_x ** 2

        # Check the result of power operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_power_backward(self):
        # Test backward propagation of gradients for power operation
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = x ** 2
        torch_y = torch_x ** 2

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestExponential(unittest.TestCase):
    def test_tensor_exp(self):
        # Test the exponential operation
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.exp(x)
        torch_y = torch_x.exp()

        # Check the result of exponential operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_exp_backward(self):
        # Test backward propagation of gradients for exponential operation
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.exp(x)
        torch_y = torch_x.exp()

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestLogarithm(unittest.TestCase):
    def test_tensor_log(self):
        # Test the logarithm operation
        x = Tensor(np.array([1.0, 2.0, np.e]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.log(x)
        torch_y = torch_x.log()

        # Check the result of logarithm operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_log_backward(self):
        # Test backward propagation of gradients for logarithm operation
        x = Tensor(np.array([1.0, 2.0, np.e]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.log(x)
        torch_y = torch_x.log()

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestLinear(unittest.TestCase):
    def test_tensor_linear(self):
        # Test the linear operation
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=False)
        weight = Tensor(np.array([[1.0, 0.0, 2.0], [0.0, -1.0, -0.5]]), requires_grad=False)
        bias = Tensor(np.array([1.0, 1.0]), requires_grad=False)

        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_weight = torch.tensor(weight.data, requires_grad=False)
        torch_bias = torch.tensor(bias.data, requires_grad=False)

        y = F.linear(x, weight, bias)
        torch_y = torch_F.linear(torch_x, torch_weight, torch_bias)

        # Check the result of linear operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_linear_backward(self):
        # Test backward propagation of gradients for linear operation
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
        weight = Tensor(np.array([[1.0, 0.0, 2.0], [0.0, -1.0, -0.5]]), requires_grad=True)
        bias = Tensor(np.array([1.0, 1.0]), requires_grad=True)

        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_weight = torch.tensor(weight.data, requires_grad=True)
        torch_bias = torch.tensor(bias.data, requires_grad=True)

        y = F.linear(x, weight, bias)
        torch_y = torch_F.linear(torch_x, torch_weight, torch_bias)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_equal(weight.grad, torch_weight.grad.numpy())
        np.testing.assert_array_equal(bias.grad, torch_bias.grad.numpy())


class TestRelu(unittest.TestCase):
    def test_tensor_relu(self):
        # Test the ReLU operation
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.relu(x)
        torch_y = torch_x.relu()

        # Check the result of ReLU operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_relu_backward(self):
        # Test backward propagation of gradients for ReLU operation
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.relu(x)
        torch_y = torch_F.relu(torch_x)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestTanh(unittest.TestCase):
    def test_tensor_tanh(self):
        # Test the Tanh operation
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.tanh(x)
        torch_y = torch_x.tanh()

        # Check the result of Tanh operation
        np.testing.assert_array_almost_equal(y.data, torch_y.numpy())

    def test_tensor_tanh_backward(self):
        # Test backward propagation of gradients for Tanh operation
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.tanh(x)
        torch_y = torch_F.tanh(torch_x)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_almost_equal(x.grad, torch_x.grad.numpy())


class TestSigmoid(unittest.TestCase):
    def test_tensor_sigmoid(self):
        # Test the sigmoid operation
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.sigmoid(x)
        torch_y = torch_x.sigmoid()

        # Check the result of sigmoid operation
        np.testing.assert_array_almost_equal(y.data, torch_y.numpy())

    def test_tensor_sigmoid_backward(self):
        # Test backward propagation of gradients for sigmoid operation
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.sigmoid(x)
        torch_y = torch_F.sigmoid(torch_x)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_almost_equal(x.grad, torch_x.grad.numpy())


class TestSoftmax(unittest.TestCase):
    def test_tensor_softmax(self):
        # Test the softmax operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.softmax(x, axis=1)
        torch_y = torch_x.softmax(dim=1)

        # Check the result of softmax operation
        np.testing.assert_array_almost_equal(y.data, torch_y.numpy())

    def test_tensor_softmax_backward(self):
        # Test backward propagation of gradients for softmax operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.softmax(x, axis=1)
        torch_y = torch_F.softmax(torch_x, dim=1)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_almost_equal(x.grad, torch_x.grad.numpy())


class TestConv2d(unittest.TestCase):
    def test_tensor_conv2d(self):
        # Test the 2D convolution operation
        x = Tensor(np.random.randn(1, 3, 5, 5), requires_grad=False)  # Input tensor (batch_size, channels, height, width)
        weight = Tensor(np.random.randn(2, 3, 3, 3), requires_grad=False)  # Weight tensor (out_channels, in_channels, kernel_height, kernel_width)
        bias = Tensor(np.random.randn(2), requires_grad=False)  # Bias tensor (out_channels)

        torch_x = torch.tensor(x.data, requires_grad=False)
        torch_weight = torch.tensor(weight.data, requires_grad=False)
        torch_bias = torch.tensor(bias.data, requires_grad=False)

        y = F.conv2d(x, weight, bias)
        torch_y = torch_F.conv2d(torch_x, torch_weight, torch_bias, stride=1, padding=0, dilation=1)

        # Check the result of convolution operation
        np.testing.assert_array_almost_equal(y.data, torch_y.numpy(), decimal=6)
    def test_tensor_conv2d_backward(self):
        # Test backward propagation of gradients for 2D convolution
        x = Tensor(np.random.randn(1, 3, 5, 5), requires_grad=True)  # Input tensor (batch_size, channels, height, width)
        weight = Tensor(np.random.randn(2, 3, 3, 3), requires_grad=True)  # Weight tensor (out_channels, in_channels, kernel_height, kernel_width)
        bias = Tensor(np.random.randn(2), requires_grad=True)  # Bias tensor (out_channels)

        torch_x = torch.tensor(x.data, requires_grad=True)
        torch_weight = torch.tensor(weight.data, requires_grad=True)
        torch_bias = torch.tensor(bias.data, requires_grad=True)

        y = F.conv2d(x, weight, bias)
        torch_y = torch_F.conv2d(torch_x, torch_weight, torch_bias, stride=1, padding=0, dilation=1)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_almost_equal(x.grad, torch_x.grad.numpy())
        np.testing.assert_array_almost_equal(weight.grad, torch_weight.grad.numpy())
        np.testing.assert_array_almost_equal(bias.grad, torch_bias.grad.numpy())


class TestDropout(unittest.TestCase):
    def test_tensor_dropout(self):
        # Test the dropout operation
        np.random.seed(42)  # For reproducibility
        torch.manual_seed(42)  # For reproducibility in PyTorch
        x = Tensor(np.random.rand(1000, 1000), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        p = 0.5

        y = F.dropout(x, p=p, training=True)
        torch_y = torch_F.dropout(torch_x, p=p, training=True)

        # Check the proportion of zero elements in the output is approximately equal to the torch version
        np.testing.assert_almost_equal(np.mean(y.data == 0), np.mean(torch_y.numpy() == 0), decimal=2)

    def test_tensor_dropout_backward(self):
        # Test backward propagation of gradients for dropout
        x = Tensor(np.random.rand(1000, 1000), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        p = 0.5

        y = F.dropout(x, p=p, training=True)
        torch_y = torch_F.dropout(torch_x, p=p, training=True)

        # Sum the output to reduce it to a scalar
        torch_y_sum = torch_y.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y_sum.backward()

        # Get the dropout masks
        y_mask = y.data != 0
        torch_mask = torch_y.detach().numpy() != 0
        combined_mask = y_mask & torch_mask

        # Ensure that the gradients are only compared for non-zero elements
        np.testing.assert_array_almost_equal(x.grad[combined_mask], torch_x.grad[combined_mask].numpy())


class TestSum(unittest.TestCase):
    def test_tensor_sum(self):
        # Test the sum operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.sum(x)
        torch_y = torch_x.sum()

        # Check the result of sum operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())

    def test_tensor_sum_backward(self):
        # Test backward propagation of gradients for sum operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.sum(x)
        torch_y = torch_x.sum()

        # Call backward() to calculate gradients
        y.backward()
        torch_y.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestMean(unittest.TestCase):
    def test_tensor_mean(self):
        # Test the mean operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        torch_x = torch.tensor(x.data, requires_grad=False)

        y = F.mean(x)
        torch_y = torch_x.mean()

        # Check the result of mean operation
        np.testing.assert_array_equal(y.data, torch_y.numpy())


    def test_tensor_mean_backward(self):
        # Test backward propagation of gradients for mean operation
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)

        y = F.mean(x)
        torch_y = torch_x.mean()

        # Call backward() to calculate gradients
        y.backward()
        torch_y.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_equal(x.grad, torch_x.grad.numpy())


class TestBinaryCrossEntropy(unittest.TestCase):
    def test_tensor_bce(self):
        # Test binary cross-entropy loss
        y_true = Tensor(np.array([1.0, 0.0, 1.0]), requires_grad=False)
        y_pred = Tensor(np.array([0.9, 0.1, 0.8]), requires_grad=False)

        torch_y_true = torch.tensor(y_true.data, requires_grad=False)
        torch_y_pred = torch.tensor(y_pred.data, requires_grad=False)

        loss = F.binary_cross_entropy(y_pred, y_true)
        torch_loss = nn.BCELoss()(torch_y_pred, torch_y_true)

        # Check the result of BCE loss
        np.testing.assert_array_almost_equal(loss.data, torch_loss.item())

    def test_tensor_bce_backward(self):
        # Test backward propagation of gradients for binary cross-entropy loss
        y_true = Tensor(np.array([1.0, 0.0, 1.0]), requires_grad=False)
        y_pred = Tensor(np.array([0.9, 0.1, 0.8]), requires_grad=True)
        torch_y_true = torch.tensor(y_true.data, requires_grad=False)
        torch_y_pred = torch.tensor(y_pred.data, requires_grad=True)

        loss = F.binary_cross_entropy(y_pred, y_true)
        torch_loss = torch_F.binary_cross_entropy(torch_y_pred, torch_y_true)

        # Call backward() to calculate gradients
        loss.backward()
        torch_loss.backward()

        # Check that gradients are computed correctly
        np.testing.assert_array_almost_equal(y_pred.grad, torch_y_pred.grad.numpy())


class TestCrossEntropy(unittest.TestCase):
    def test_tensor_cross_entropy(self):
        # Test cross-entropy loss
        y_true = Tensor(np.array([2, 1, 0]), requires_grad=False, dtype=np.float64)  # True class indices
        y_pred = Tensor(np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [2.0, 0.1, 1.0]]), requires_grad=False, dtype=np.float64)

        torch_y_true = torch.tensor(y_true.data, dtype=torch.long)
        torch_y_pred = torch.tensor(y_pred.data, dtype=torch.float64)

        # Compute loss using custom cross-entropy function
        softmax_y_pred = F.softmax(y_pred)  # Convert predictions to probabilities

        loss = F.cross_entropy(softmax_y_pred, y_true)
        torch_loss = nn.CrossEntropyLoss()(torch_y_pred, torch_y_true)

        # Check the result of cross-entropy loss
        np.testing.assert_array_almost_equal(loss.data, torch_loss.item(), decimal=6)

    def test_tensor_cross_entropy_backward(self):
        # Test cross-entropy loss
        y_true = Tensor(np.array([2, 1, 0]), requires_grad=False, dtype=np.float64)  # True class indices
        y_pred = Tensor(np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [2.0, 0.1, 1.0]]), requires_grad=True, dtype=np.float64)
        torch_y_true = torch.tensor(y_true.data, requires_grad=False, dtype=torch.long)
        torch_y_pred = torch.tensor(y_pred.data, requires_grad=True, dtype=torch.float64)

        # Compute loss using custom cross-entropy function
        softmax_y_pred = F.softmax(y_pred)  # Convert predictions to probabilities

        loss = F.cross_entropy(softmax_y_pred, y_true)
        torch_loss = torch_F.cross_entropy(torch_y_pred, torch_y_true)

        # Call backward() to calculate gradients
        loss.backward()
        torch_loss.backward()

        # Check the result of cross-entropy loss
        np.testing.assert_array_almost_equal(y_pred.grad, torch_y_pred.grad.numpy())


if __name__ == "__main__":
    unittest.main()
