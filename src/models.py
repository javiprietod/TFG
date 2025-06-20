import torch
from torch import nn


class LogisticModel(nn.Module):
    """Simple feed-forward logistic regression style model."""

    def __init__(self, input_dim: int, hidden_sizes: list[int]):
        """Create the network architecture.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_sizes : list[int]
            Sizes of the hidden layers.
        """

        super().__init__()
        self.layers = []
        self.input_dim = input_dim

        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
                self.layers.append(nn.Sigmoid())
                # self.layers.append(nn.Dropout(0.2))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                self.layers.append(nn.Sigmoid())
                # self.layers.append(nn.Dropout(0.2))

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_dim, 2))
        else:
            self.layers.append(nn.Linear(hidden_sizes[-1], 2))
        self.layers.append(nn.Softmax(dim=1))  # revisar si lo hace bien
        self.model = nn.Sequential(*self.layers)

    def forward(self, inputs):
        """Run a forward pass of the network."""

        return self.model(inputs)
