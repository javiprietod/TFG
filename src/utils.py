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
        self.columns: np.ndarray = None  # original columns
        self.good_class: str = None
        self.id_column: str = None
        self.target_column: str = None
        self.batch_size: int = None
        self.changeable_col_names: list[str] = None
        self.cols_for_mask: np.ndarray = None
        self.scaler: StandardScaler = None
        self.cols_for_scaler: torch.Tensor = None
        self.cols_for_scaler_names: np.ndarray = None
        self.mean_scaled: torch.Tensor = None
        self.dx_scaled: torch.Tensor = None
        self.obj_cols: dict[str, list[str]] = None  # values in categorical columns
        self.int_cols: torch.Tensor = None
        self.max_values: torch.Tensor = None
        self.min_values: torch.Tensor = None
        self.data: pd.DataFrame = None
        self.threshold: float = 0.5 + 1e-5


def clean_data(
    df: pd.DataFrame, metadata: DatasetMetadata, scale: True
) -> pd.DataFrame:
    """
    This function cleans the data by removing the rows with missing values.

    Args:
        df: dataframe with the data.

    Returns:
        dataframe without missing values.
    """

    # drop missing values
    df = df.dropna()

    df = df.drop(metadata.id_column, axis=1) if metadata.id_column else df

    # drop columns that only have one value
    df = df.loc[:, df.nunique() > 1]

    # drop duplicates
    df = df.drop_duplicates()

    # scale the numerical columns
    scaler = StandardScaler()

    int_cols = (
        df.drop(metadata.target_column, axis=1).select_dtypes(include=[int]).columns
    )
    num_columns = df.select_dtypes(exclude=["object"]).columns.drop(
        metadata.target_column
    )
    df[num_columns] = scaler.fit_transform(df[num_columns])
    metadata.cols_for_scaler_names = num_columns
    # one hot encode the categorical columns
    obj_columns = df.select_dtypes(include=[object]).columns

    metadata.obj_cols = {col: df[col].unique() for col in obj_columns}

    df_encoded = pd.get_dummies(df, columns=obj_columns, drop_first=True)

    bool_cols = df_encoded.select_dtypes(include=[bool]).columns

    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    metadata.scaler = scaler
    metadata.cols_for_scaler = torch.tensor(
        df_encoded.drop(metadata.target_column, axis=1).columns.isin(num_columns)
    )
    metadata.int_cols = torch.tensor(
        df_encoded.drop(metadata.target_column, axis=1).columns.isin(int_cols)
    )

    instance_example = torch.tensor(df_encoded.drop(metadata.target_column, axis=1).iloc[0].values)
    mean_scaled = torch.zeros_like(instance_example, dtype=torch.float32)
    mean_scaled[metadata.cols_for_scaler == 1] = (
        torch.tensor(metadata.scaler.mean_).to(torch.float)
    )
    dx_scaled = torch.ones_like(instance_example, dtype=torch.float32)
    dx_scaled[metadata.cols_for_scaler == 1] = (
        torch.tensor(metadata.scaler.scale_).to(torch.float)
    )
    metadata.mean_scaled = mean_scaled
    metadata.dx_scaled = dx_scaled

    return df_encoded


def transform_onehot(raw_instance, original_columns, final_columns):
    encoded_instance = {col: 0 for col in final_columns}

    # Handle numerical columns (those in both original and final directly)
    for col in original_columns:
        if col in final_columns and col in raw_instance:
            encoded_instance[col] = raw_instance[col]

    # Handle one-hot encoded fields
    for col in final_columns:
        if "_" in col:
            base_col, val = col.split("_", 1)
            if base_col in raw_instance:
                raw_val = str(raw_instance[base_col])
                if raw_val == val:
                    encoded_instance[col] = 1

    return torch.tensor(list(encoded_instance.values()))


def transform_onehot_inverse(
    df: pd.DataFrame, metadata: DatasetMetadata
) -> pd.DataFrame:
    result = df.copy()
    restored_cols = {}

    for col, values in metadata.obj_cols.items():
        # Get the dummy column names
        dummy_cols = [c for c in df.columns if c.startswith(f"{col}_")]

        # Create a column to store the restored category
        def get_original_value(row):
            for i, dummy_col in enumerate(dummy_cols):
                if row.get(dummy_col, 0) == 1:
                    return values[i + 1]  # Match found
            return values[0]  # If none are 1, it's the dropped first value

        # Optionally, drop the dummy columns
        result = result.drop(columns=[col for col in dummy_cols if col in df.columns])
        result[col] = df[dummy_cols].apply(get_original_value, axis=1)

    # Add the restored columns
    for col, series in restored_cols.items():
        result[col] = series
    return result


def clean_instance(instance: dict, metadata: DatasetMetadata) -> torch.Tensor:
    """
    This function cleans the data by removing the rows with missing values.

    Args:
        df: dataframe with the data.

    Returns:
        dataframe without missing values.
    """

    instance = transform_onehot(instance, instance.keys(), metadata.columns)

    unscaled_cols = metadata.scaler.transform(
        instance[metadata.cols_for_scaler].reshape(1, -1)
    )
    instance[metadata.cols_for_scaler] = torch.tensor(
        unscaled_cols, dtype=torch.float32
    )

    return instance


def load_data(
    path: str, index: int = None, batch_size: int = 1024, get_sample: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, DatasetMetadata]:
    set_seed(42)

    df = pd.read_csv(path)

    metadata = DatasetMetadata()

    metadata.path = path
    metadata.batch_size = batch_size
    with open("datasets.yaml", "r") as file:
        datasets = yaml.safe_load(file)[metadata.path]

    id_column = datasets["id_column"]
    target_column = datasets["target_column"]

    # target column
    # search for the column that has only 1 and 0
    metadata.id_column = id_column
    metadata.target_column = target_column
    metadata.good_class = datasets["good_class"]
    # class_weights = torch.tensor(list(df[target_column].value_counts(normalize=True))[::-1])
    class_weights = torch.tensor([0.2, 0.8])

    data = clean_data(df, metadata, target_column)

    if get_sample:
        distinct_outputs = sorted(data[target_column].unique())
        sample0 = data[data[target_column] == distinct_outputs[0]].iloc[0:1]
        sample1 = data[data[target_column] == distinct_outputs[1]].iloc[0:1]
        samples = pd.concat([sample0, sample1])
        samples.reset_index(drop=True, inplace=True)
    elif index is not None:
        data = data.iloc[index : index + 1]

    metadata.columns = data.drop(target_column, axis=1).columns

    with open("datasets.yaml", "r") as file:
        datasets = yaml.safe_load(file)

    cols_for_mask = datasets[path]["weights"]
    metadata.changeable_col_names = cols_for_mask
    # create a mask for the columns
    col_mask = data.drop(target_column, axis=1).columns.isin(cols_for_mask)
    metadata.cols_for_mask = col_mask

    # Max and min values for the columns
    metadata.max_values = torch.tensor(
        data.drop(target_column, axis=1).max().values, dtype=torch.float32
    )
    metadata.min_values = torch.tensor(
        data.drop(target_column, axis=1).min().values, dtype=torch.float32
    )

    if get_sample:
        return samples, metadata

    elif index is None:
        dataset = LoanDataset(data, target_column)

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42)
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_weights,
            metadata,
        )

    else:
        return torch.tensor(data.loc[0].values), metadata


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

