import torch
from torch import nn

class LogisticModel(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.layers = []
        

        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
                self.layers.append(nn.ReLU())
                # self.layers.append(nn.Dropout(0.2))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                self.layers.append(nn.ReLU())
                # self.layers.append(nn.Dropout(0.2))

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_dim, 2))
        else:
            self.layers.append(nn.Linear(hidden_sizes[-1], 2))
        self.model = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.model(inputs)

        
