
# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# other libraries
import numpy as np
import pandas as pd
import os
import random
import yaml

# TODO : Change doc strings 
class LoanDataset(Dataset):
    """
    This class is the dataset loading the data.

    Attr:
        dataset: tensor with all the prices data. Dimensions:
            [number of days, 24].
        past_days: length used for predicting the next value.
    """

    dataset: torch.Tensor

    def __init__(self, dataset: pd.DataFrame, target_column: str) -> None:
        """
        Constructor of ElectricDataset.

        Args:
            dataset: dataset in dataframe format. It has three columns
                (price, feature 1, feature 2) and the index is
                Timedelta format.
            past_days: number of past days to use for the
                prediction.
        """

        self.data = torch.tensor(np.array(dataset.drop(target_column, axis=1)))
        self.targets = torch.tensor(np.array(dataset[target_column]))

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            number of days in the dataset.
        """

        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        
        This method returns an element from the dataset based on the
        index. It only has to return the prices.

        Args:
            index: index of the element.

        Returns:
            past values, starting to collecting those in the zero
                index. Dimensions: [sequence length, 24].
            current values. Start to collect those in the index
                self.sequence. Dimensions: [24].
        """
        
        return self.data[index], self.targets[index]
    
class DatasetMetadata:
    def __init__(self):
        self.path: str = None
        self.batch_size: int = None
        self.cols_for_mask: np.ndarray = None
        self.scaler: StandardScaler = None
        self.cols_for_scaler: np.ndarray = None
        self.int_cols: np.ndarray = None
        self.columns: np.ndarray = None
        self.data: pd.DataFrame = None


def clean_data(df: pd.DataFrame, metadata: DatasetMetadata) -> pd.DataFrame:
    """
    This function cleans the data by removing the rows with missing values.

    Args:
        df: dataframe with the data.

    Returns:
        dataframe without missing values.
    """

    # drop missing values
    df = df.dropna()
    # filter for the columns that have different values for each row
    # take the first column of the filtered columns
    # This can change from dataset to dataset but it is a good starting point
    id_column = df.loc[:, df.nunique() == len(df)].columns[0]
    
    df = df.drop(id_column, axis=1)

    # drop columns that only have one value
    df = df.loc[:, df.nunique() > 1]

    # drop duplicates
    df = df.drop_duplicates()

    # scale the numerical columns
    scaler = StandardScaler()
    target_column = df.loc[:, df.nunique() == 2].select_dtypes(include=[int]).columns[-1]
    int_cols = df.drop(target_column, axis=1).select_dtypes(include=[int]).columns
    num_columns = df.select_dtypes(exclude=['object']).columns.drop(target_column)
    df[num_columns] = scaler.fit_transform(df[num_columns])
    # one hot encode the categorical columns
    obj_columns = df.select_dtypes(include=[object]).columns

    df_encoded = pd.get_dummies(df, columns=obj_columns, drop_first=True)
    
    bool_cols = df_encoded.select_dtypes(include=[bool]).columns

    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    metadata.scaler = scaler
    metadata.cols_for_scaler = df_encoded.drop(target_column, axis=1).columns.isin(num_columns)
    metadata.int_cols = df_encoded.drop(target_column, axis=1).columns.isin(int_cols)

    return df_encoded

def load_data(path: str, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, DatasetMetadata]:
    
    set_seed(42)

    df = pd.read_csv(path)

    metadata = DatasetMetadata()

    metadata.path = path
    metadata.batch_size = batch_size

    # target column
    # search for the column that has only 1 and 0
    target_column = df.loc[:, df.nunique() == 2].select_dtypes(include=[int]).columns[0]
    # class_weights = torch.tensor(list(df[target_column].value_counts(normalize=True))[::-1])
    class_weights = torch.tensor([0.2, 0.8])

    data = clean_data(df, metadata)
    # print(len(data))
    # class imbalance

    metadata.columns = data.drop(target_column, axis=1).columns

    with open("datasets.yaml", "r") as file:
        datasets = yaml.safe_load(file)

    cols_for_mask = datasets[path]
    # create a mask for the columns
    col_mask = data.columns.isin(cols_for_mask)[:-1]
    metadata.cols_for_mask = col_mask
    metadata.data = data

    dataset = LoanDataset(data, target_column)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, class_weights, metadata


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed number to fix radomness
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None

@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: pytorch model.
    """

    # transform model to double
    for param in model.parameters():
        param.data = param.data.double()