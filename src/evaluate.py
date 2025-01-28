import torch
from torch.utils.data import DataLoader

from src.train_functions import test_step
from src.utils import load_data, load_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(path: str, name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO

    # load data
    test_data: DataLoader
    _, _, test_data, _, _ = load_data(path, batch_size=1024)

    # define name and writer

    # define model
    model = load_model(f"{name}").to(device)

    # call test step and evaluate accuracy
    accuracy, f1_score, confusion_matrix = test_step(model, test_data, device)

    return accuracy, f1_score, confusion_matrix


if __name__ == "__main__":
    accuracy, f1_score, confusion_matrix = main('data/Loan_default_3.csv', 'model_logistic_lr_0.0004_bs_128_hs_[64]_50')
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")
    print(f"Confusion Matrix: \n{confusion_matrix}")
