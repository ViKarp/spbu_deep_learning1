import torch
from typing import List, Tuple

class NAGOptimizer:
    def __init__(self, parameters: torch.Tensor, lr: float = 0.01, momentum: float = 0.9) -> None:
        """
        Initializes the NAG optimizer.

        Args:
            parameters (list[torch.Tensor]): List of model parameters (weights and biases).
            lr (float): Learning rate. Default is 0.01.
            momentum (float): Momentum coefficient. Default is 0.9.
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = torch.zeros_like(parameters)

    def step(self) -> None:
        """
        Performs one optimization step using Nesterov Accelerated Gradient (NAG)
        for each parameter.
        """
        with torch.no_grad():
            v_prev = self.velocities.clone()
            self.parameters -= self.momentum * v_prev

            grad = self.parameters.grad if self.parameters.grad is not None else torch.zeros_like(self.parameters)
            self.velocities = self.momentum * v_prev + self.lr * grad
            self.parameters -= self.velocities

    def zero_grad(self) -> None:
        """
        Resets the gradients for all parameters to zero.
        """
        if self.parameters.grad is not None:
            self.parameters.grad.detach_()
            self.parameters.grad.zero_()

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Compute the sigmoid activation function."""
    return 1 / (1 + torch.exp(-x))

def negative_log_likelihood(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log likelihood loss.

    Parameters:
    predictions (torch.Tensor): Predicted probabilities.
    labels (torch.Tensor): True labels.

    Returns:
    torch.Tensor: Computed negative log likelihood.
    """
    predictions = torch.clamp(predictions, min=1e-10, max=1-1e-10)
    nll = -torch.mean(labels * torch.log(predictions) + (1 - labels) * torch.log(1 - predictions))
    return nll

def train_neuron_with_NAG(features: List[List[float]],
                 labels: List[float],
                 initial_weights: List[float],
                 initial_bias: float,
                 learning_rate: float,
                 epochs: int) -> Tuple[List[float], float, List[float]]:
    """
    Train a simple neuron using gradient descent.

    Parameters:
    features (List[List[float]]): Input feature data.
    labels (List[float]): True output labels.
    initial_weights (List[float]): Initial weights for the neuron.
    initial_bias (float): Initial bias for the neuron.
    learning_rate (float): Learning rate for weight updates.
    epochs (int): Number of training iterations.

    Returns:
    Tuple[List[float], float, List[float]]: Updated weights, updated bias, and a list of negative log likelihood (NLL) values per epoch.
    """
    weights = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)
    bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    nll_values = []
    optimizer = NAGOptimizer(weights, lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        optimizer.zero_grad()

        linear_output = torch.matmul(features, weights) + bias
        predictions = sigmoid(linear_output)

        nll = negative_log_likelihood(predictions, labels)
        nll_values.append(round(nll.item(), 4))

        nll.backward()

        optimizer.step()

    updated_weights = torch.round(weights, decimals=4).tolist()
    updated_bias = round(bias.item(), 4)

    return updated_weights, updated_bias, nll_values


features = [
    [0.5, 1.5, 2.0],
    [1.0, 2.0, 3.0],
    [1.5, 2.5, 3.5],
    [2.0, 1.0, 0.5],
    [-1.0, -1.5, -2.0],
    [-1.5, -2.0, -2.5],
    [0.0, 0.5, 1.0],
    [2.0, 3.0, 4.0],
    [-2.0, -2.5, -3.0],
    [3.0, 4.0, 5.0]
]
labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]
initial_weights = [0.2, -0.1, 0.4]
initial_bias = 0.0
learning_rate = 0.05
epochs = 50

updated_weights, updated_bias, nll_values = train_neuron_with_NAG(features, labels, initial_weights, initial_bias, learning_rate, epochs)

print("Updated weights:", updated_weights)
print("Updated bias:", updated_bias)
print("NLL values per epoch:", nll_values)
