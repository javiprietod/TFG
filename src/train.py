import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import importlib
import src.train_functions
import src.utils
import src.models

importlib.reload(src.train_functions)
importlib.reload(src.utils)
importlib.reload(src.models)

from src.train_functions import train_step, val_step
from src.utils import load_data, set_seed, save_model, parameters_to_double
from src.models import LogisticModel


set_seed(42)

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program for training.
    """
    
    # probar con más epochs los mejores 3 modelos

    # hyperparameters
    epochs: int = 5
    lr: float = 3e-3
    batch_size: int = 1024
    # hidden_sizes: tuple[int, ...] = (512, 256, 128, 64) #256 128

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(df_encoded, batch_size=batch_size)

    # define name and writer
    name: str = f"model_logistic_lr_{lr}_bs_{batch_size}_epochs_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = LogisticModel(
        inputs.shape[1],
    ).to(device)
    parameters_to_double(model)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.BCELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
