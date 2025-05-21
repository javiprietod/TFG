import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import importlib
import src.train_functions
import src.utils
import src.models
import src.evaluate

importlib.reload(src.train_functions)
importlib.reload(src.utils)
importlib.reload(src.models)
importlib.reload(src.evaluate)

from src.train_functions import train_step, val_step
from src.utils import load_data, set_seed, save_model
from src.models import LogisticModel
from src.evaluate import main as main_ev


set_seed(42)

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(path: str, model_name: str = None) -> None:
    """
    This function is the main program for training.
    """

    # hyperparameters
    with open("train.yaml", "r") as f:
        config = yaml.safe_load(f)
    name: str = config["name"] if model_name is None else model_name
    epochs: int = int(config["epochs"])
    lr: float = float(config["lr"])
    batch_size: int = int(config["batch_size"])
    hidden_sizes: tuple[int, ...] = tuple(config["hidden_sizes"])
    weight_decay: float = float(config["weight_decay"])

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _, class_weights, _ = load_data(path, batch_size=batch_size)

    # define name and writer
    # name: str = f"model_logistic_lr_{lr}_bs_{batch_size}_hs_{hidden_sizes}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = LogisticModel(
        inputs.shape[1],
        hidden_sizes,
    ).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

    # save model
    save_model(model, name)

    # TODO: quitar
    accuracy, f1_score, confusion_matrix = main_ev(path, name)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")
    print(f"Confusion Matrix: \n{confusion_matrix}")
    return model


if __name__ == "__main__":
    main("data/Loan_default_2.csv", "model_small_2")
