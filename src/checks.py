import importlib
import src.utils
import src.models
import src.counterfactual

importlib.reload(src.utils)
importlib.reload(src.models)
importlib.reload(src.counterfactual)

from src.utils import DatasetMetadata
from src.counterfactual import unscale_instance, scale_instance, unscale_batch, scale_batch, distance, newton_op
from src.models import LogisticModel

import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Checks:
    """Collection of post-hoc checks for generated counterfactuals."""
    def __init__(self, model: LogisticModel, metadata: DatasetMetadata, reg_int: bool = False, reg_clamp: bool = False, noise: float = 1e-3, n_points: int = 10000, noise_int: float = 2.5, dataset: torch.Tensor = None):
        
        self.model = model
        self.metadata = metadata
        self.dataset = dataset
        self.reg_int = reg_int
        self.reg_clamp = reg_clamp
        self.noise = noise
        self.n_points = n_points
        self.noise_int = noise_int
        self.distance_threshold = 1e-4
        self.sorted_points = None
        self.diff_factors = []
        self.model.eval()


    def __call__(self, person: torch.Tensor, person_new: torch.Tensor, weights: torch.Tensor):
        """
        Check if the new person is a valid counterfactual.
        """
        # Check if the new person is a valid counterfactual
        if not self.validity_check(person, person_new):
            print("The new person is not a valid counterfactual.")
            return False
        
        # Check if the new person is plausible
        if self.reg_clamp and not self.plausibility_check(person_new):
            print("The new person is not plausible.")
            return False

        # Check if the new person is minimal
        if self.reg_int:
            valid, sorted_points = self.integer_minimality_check(person, person_new, weights)
            if not valid:
                print("The new person is not integer minimal.", "The length of the sorted points is: ", len(sorted_points))
                self.sorted_points = sorted_points
                return False

        else:
            if sum(weights != 0) < 32:
                valid, sorted_points = self.minimality_check(person, person_new, weights)
                if not valid:
                    print("The new person is not minimal.", "The length of the sorted points is: ", len(sorted_points))
                    self.sorted_points = sorted_points
                    return False
                
        if not self.outlier_ckeck(self.dataset, person_new):
            print("The new person is an outlier.")
            return False
        
        if not self.stability_check_global(person, person_new, weights):
            print("The model is not stable around this point")
        
        return True


    def minimality_check(
        self,
        person: torch.Tensor,
        person_new: torch.Tensor,
        weights: torch.Tensor,
        ):
        """Check minimality of a continuous counterfactual candidate."""
        ranges = [
            np.linspace(
            person_new[i].item() - 0.01 * (self.metadata.max_values[i] - self.metadata.min_values[i]),
            person_new[i].item() + 0.01 * (self.metadata.max_values[i] - self.metadata.min_values[i]),
            5
            )
            if weights[i].item() != 0 else [person_new[i].item()]
            for i in range(person_new.shape[0])
        ]
        grid = np.array(np.meshgrid(*ranges)).T.reshape(-1, person_new.shape[0])
        points = torch.tensor(grid, dtype=torch.float32).to(device)
        points = (
            torch.clamp(points, self.metadata.min_values, self.metadata.max_values)
            if self.reg_clamp
            else points
        )
        outputs = self.model(points)
        # pandas dataset
        b = pd.DataFrame(points, columns=self.metadata.columns)
        b["output"] = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        distances = torch.tensor([distance(person, p, weights) for p in points])
        b["distance"] = distances.detach().cpu().numpy()

        d = distance(person, person_new, weights).item()
        sorted_b = b[b["distance"] < (d - 1e-7)][b["output"] == 0].sort_values(by="distance")
        # sorted_b = sorted_b[(sorted_b["distance"] - d) < self.distance_threshold]
        return len(sorted_b) == 0, sorted_b

    def integer_minimality_check(
        self,
        person: torch.Tensor,
        person_new_int: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Check minimality for counterfactuals with integer features."""
        w = ((weights != 0) & ~self.metadata.int_cols) * weights
        noise_tensor = np.random.uniform(
            -self.noise, self.noise, (self.n_points, person_new_int.shape[0])
        ) * w.numpy()
        rounded_noise_tensor = np.random.randint(
            -self.noise_int, self.noise_int, (self.n_points, person_new_int.shape[0])
        ) * (((weights != 0) & self.metadata.int_cols) * weights).numpy()

        points = scale_batch(
            torch.tensor(
                noise_tensor
                + rounded_noise_tensor
                + unscale_instance(person_new_int, self.metadata).detach().cpu().numpy().reshape(-1),
                dtype=torch.float32,
            ),
            self.metadata,
        ).to(device)
        
        points = (
            torch.clamp(points, self.metadata.min_values, self.metadata.max_values)
            if self.reg_clamp
            else points
        )

        outputs = self.model(points)

        points_unscaled = unscale_batch(points, self.metadata)
        b = pd.DataFrame(points_unscaled, columns=self.metadata.columns)

        # add person_new_int to the dataframe
        b["output"] = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        distances = torch.tensor([distance(person, p, weights) for p in points])
        b["distance"] = distances.detach().cpu().numpy()

        d = distance(person, person_new_int, ((weights != 0) & ~self.metadata.int_cols) * weights).item()
        
        sorted_b = b[b["distance"] < (d - 1e-7)][b["output"] == 0].sort_values(by="distance")
        return len(sorted_b) == 0, sorted_b
    
    def validity_check(
      self,
      person: torch.Tensor,
      person_new: torch.Tensor,
    ):
        """Return True if person_new changes the model prediction."""
        return (
            (self.model(person_new.unsqueeze(0))[0][self.metadata.good_class].item() >= 0.5)
            != (self.model(person.unsqueeze(0))[0][self.metadata.good_class].item() >= 0.5)
        )
    
    def plausibility_check(
        self,
        person_new: torch.Tensor,
    ):
        """Check whether the new instance lies inside the allowed domain."""
        return (
            torch.clamp(
                person_new,
                self.metadata.min_values,
                self.metadata.max_values,
            )
            == person_new
        ).all().item()
    
    def outlier_ckeck(
        self,
        dataset: torch.Tensor,
        person_new: torch.Tensor,
    ):
        """
        Check if the new person is an outlier in the dataset with Local Outlier Factor (LOF).
        Uses the Local Outlier Factor (LOF) algorithm to detect outliers in the dataset.
        Args:
            dataset (pd.DataFrame): The dataset to check against.
            person_new (torch.Tensor): The new person to check.
        Returns:
            bool: True if the new person is not an outlier, False otherwise.
        """
        if dataset is None:
            return True
        data = dataset.numpy()
        # Add the new person to the dataset
        data = np.vstack([data, person_new.detach().cpu().numpy()])
        # Fit the Local Outlier Factor model
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outliers = lof.fit_predict(data)
        # Check if the new person is an outlier
        return outliers[-1] == 1
    
    def stability_check_one_var(self, person, person_new, weights):
        """Assess model stability when varying one feature at a time."""

        max_diff_factor = 0
        for perc_fator in [1, 2, 5]:
            for i, scaled in enumerate(self.metadata.cols_for_scaler):
                if scaled & (weights[i] != 0):
                    for sign in [-1, 1]:
                        person_close = person.detach().clone()
                        person_close[i] += sign * (self.metadata.max_values[i] - self.metadata.min_values[i]) * perc_fator / 100
                        person_close_new, _ = newton_op(self.model, person_close, self.metadata, weights, 0.2, reg_int=self.reg_int, reg_clamp=self.reg_clamp, print_=False, der = False)
                        diff_factor = distance(person_new, person_close_new, weights) / distance(person, person_close, weights)
                        if diff_factor > max_diff_factor:
                            max_diff_factor = diff_factor
                            # print(max_diff_factor, distance(person_new, person_close_new, weights), distance(person, person_close, weights), i, perc_fator)
       
        return max_diff_factor < 2
    

    def stability_check_global(self, person, person_new, weights):
        """Assess global stability around the generated counterfactual."""

        max_diff_factor = 0
        num_points = 10
        for perc_fator in [1]:
            for sign in [-1, 1]:
                noise = (self.metadata.max_values - self.metadata.min_values) * perc_fator / 100
                points = torch.tensor(
                    np.random.uniform(-noise, noise, (num_points, person.shape[0]))
                    * (weights != 0).numpy() 
                    + person.detach().cpu().numpy().reshape(-1),
                    dtype=torch.float32,
                ).to(device)
                
                for person_close in points:

                    
                    person_close_new, _ = newton_op(self.model, person_close, self.metadata, weights, 0.2, reg_int=self.reg_int, reg_clamp=self.reg_clamp, print_=False, der = False)
                    diff_factor = distance(person_new, person_close_new, weights) / distance(person, person_close, weights)
                    max_diff_factor = max(max_diff_factor, diff_factor)
        self.diff_factors.append(max_diff_factor.item())
        return True
            