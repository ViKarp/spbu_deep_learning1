import torch


class MyBatchNorm:
    """
    Custom Batch Normalization layer.

    Args:
        num_features (int): Number of features in the input.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): Momentum for updating the running mean and variance. Default: 0.1.

    Attributes:
        gamma (torch.Tensor): Trainable scale parameter.
        beta (torch.Tensor): Trainable shift parameter.
        running_mean (torch.Tensor): Running mean used during inference.
        running_var (torch.Tensor): Running variance used during inference.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Trainable parameters
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def __call__(self, x, training=True):
        """
        Applies batch normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            training (bool): Flag to specify if the layer is in training mode. Default: True.

        Returns:
            torch.Tensor: Batch normalized tensor.
        """
        if training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        out = self.gamma * x_hat + self.beta
        return out


class MyLinear:
    """
    Custom Linear (fully connected) layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If True, includes a bias term. Default: True.

    Attributes:
        weight (torch.Tensor): Trainable weight matrix.
        bias (torch.Tensor): Trainable bias vector.
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features) * 0.01
        self.bias = torch.zeros(out_features) if bias else None

    def __call__(self, x):
        """
        Applies the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """

        output = torch.matmul(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output


class MyDropout:
    """
    Custom Dropout layer.

    Args:
        p (float): Probability of an element to be zeroed. Default: 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, training=True):
        """
        Applies dropout to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            training (bool): If True, applies dropout. Otherwise, returns the input unchanged.

        Returns:
            torch.Tensor: Tensor after dropout.
        """
        if training:
            mask = (torch.rand(x.shape) > self.p).float()
            return mask * x / (1.0 - self.p)
        else:
            return x


class MyReLU:
    """
    Custom ReLU activation function.
    """

    def __call__(self, x):
        """
        Applies the ReLU activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor where all negative values are set to 0.
        """
        return torch.max(x, torch.zeros_like(x))


class MySoftmax:
    """
    Custom Softmax activation function.
    """

    def __call__(self, x, dim=-1):
        """
        Applies the Softmax function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which Softmax will be computed. Default: -1.

        Returns:
            torch.Tensor: Tensor after applying softmax.
        """
        exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


class MySigmoid:
    """
    Custom Sigmoid activation function.
    """

    def __call__(self, x):
        """
        Applies the Sigmoid function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying sigmoid.
        """
        return 1 / (1 + torch.exp(-x))


x = torch.randn(10, 5)
bn = MyBatchNorm(num_features=5)
dropout = MyDropout(p=0.3)
linear = MyLinear(in_features=5, out_features=3)
mr = MyReLU()
sm = MySoftmax()
sg = MySigmoid()
output = bn(x, training=True)
output = linear(output)
output = dropout(output, training=True)
output = mr(output)
output = sm(output)
output = sg(output)

print(output)
