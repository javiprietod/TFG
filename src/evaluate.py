import torch
from torch.utils.data import DataLoader

from src.train_functions import test_step
from src.utils import load_data, load_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO

    # load data
    test_data: DataLoader
    _, _, test_data = load_data(df_encoded, batch_size=1024)

    # define name and writer

    # define model
    model = load_model(f"{name}").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)

    return accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('model_logistic_lr_0.003_1024_5')}")
