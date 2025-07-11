# DL From Scratch

Implementing Deep Learning models from first principles — no high‑level frameworks, just raw NumPy and education-first clarity.

# 📖 Overview

This project is a lightweight, transparent collection of deep learning components and models built entirely from scratch, using only NumPy. The goal is to demonstrate core algorithms—understand gradients, forward/backward propagation, optimizers, and network structures by directly manipulating the mathematics. This is ideal for educational purposes, for those who want to understand the inner workings of deep learning without the abstraction layers of high-level libraries.

# 🚀 Features

- **Autograd Tensors**: An automatic differentiation engine to compute gradients.
- **Core Components**: Implementations of layers, activation functions, loss functions, optimizers, and more.
- **Models**: Fully functional neural networks, including MLPs and CNNs.
- **Training**: Custom training loops with gradient descent and backpropagation.
- **No Dependencies**: Pure NumPy, no high-level libraries like TensorFlow or PyTorch.
- **Educational Focus**: Each component is designed to be clear and understandable, with detailed comments and explanations.

# 📚 Documentation

- **Installation**: Clone the repository and install dependencies.
- **Usage**: Examples of how to use the components and models.
- **Contributing**: Guidelines for contributing to the project.
- **License**: MIT License.

# 🧪 Usage Examples

Train a simple MLP on synthetic data:

```bash
python examples/train_binary_classifier.py
```

Train a CNN on MNIST:

```bash
python examples/train_digit_classifier.py
```

# 🎓 Educational Goals

- **Transparency**: Avoid black box behavior—learn by looking under the hood.
- **Flexibility**: Modify and extend components to suit your needs.
- **Understanding**: See inside activations, gradients, weight updates.

# 📂 Project Structure

```bash

DL-From-Scratch/
├── examples/                   # Example scripts for training models
├── src/                        # Source code for components and models
    ├── data/                   # Data loading and preprocessing utilities
    ├── deep_learning/          # Core deep learning components
        ├── module/             # Layer implementations (Dense, Conv2D, etc.)
        ├── loss/               # Loss functions (CrossEntropy, MSE, etc.)
        ├── optimizers/         # Optimizers (SGD, Adam, etc.)
        ├── datasets/           # Dataset classes for loading data
        ├── utils/              # Utility functions and classes
        ├── functional/         # Functional API for operations
        ├── tensor/             # Autograd tensor implementation

├── tests/                      # Unit tests for components
├── README.md                   # Project overview and documentation
```

# 🔭 Future Work
- **More Models**: Implement additional architectures like RNNs, Transformers, etc.
- **Improving Performance**: Optimize the code for speed and memory efficiency maintaining educational clarity.

# 🧩 Extend & Contribute
- Contributions, PRs, and issue reports are welcome!

# 📫 Contact
Interested in collaborating or have questions? Reach out via GitHub issues or email.

- **Author**: Arif A. Othman
- **Email**: arithman34@hotmail.com
- **GitHub**: [arithman34](https://github.com/arithman34)

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
