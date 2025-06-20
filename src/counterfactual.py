import importlib
import src.utils
import src.models
import src.train

importlib.reload(src.utils)
importlib.reload(src.models)
importlib.reload(src.train)

from src.utils import load_data, load_model, DatasetMetadata
from src.train import main as main_train
import torch
import pandas as pd
import numpy as np
# import sympy as sp

# str to sympy
# from sympy.parsing.sympy_parser import parse_expr

# from torch.utils.data import DataLoader
from src.models import LogisticModel
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = device if not torch.backends.mps.is_available() else torch.device("mps")


class State:
    """Mutable state used during the counterfactual search."""

    def __init__(self, model, metadata, max_epochs):
        self.model: LogisticModel = model
        self.metadata: DatasetMetadata = metadata
        self.epochs: int = 0
        self.max_epochs: int = max_epochs
        self.reg_int = False  # When to apply integer regularization
        self.reg_vars = False  # When to apply nº variables regularization
        self.reg_clamp = False


def distance(
    original: torch.Tensor,
    new: torch.Tensor,
    weights: torch.Tensor,
    state: State = None,
    with_sum: bool = True,
):
    """Compute a weighted squared distance between two instances."""
    # clamped_new = torch.clamp(new, scale_instance(state.metadata.min_values, state.metadata), scale_instance(state.metadata.max_values, state.metadata))
    cost = (original - new) ** 2 * weights

    """ Regularización de maximos y mínimos
    reg_term = 0
    # max(x-b, 0)
    # Si x > b, entonces añado una distancia muy grande
    # if torch.max(new - state.metadata.max_values.to(device), torch.zeros_like(new)).sum() != 0:
    #     print("Warning: some values are out of bounds")
    cost += reg_term * torch.max(
        new - state.metadata.max_values.to(device), torch.zeros_like(new)
    )
    # max(a-x, 0)
    # Si x < a, entonces añado una distancia muy grande
    # if torch.max(state.metadata.min_values.to(device) - new, torch.zeros_like(new)).sum() != 0:
    #     print("Warning: some values are out of bounds")
    cost += reg_term * torch.max(
        state.metadata.min_values.to(device) - new, torch.zeros_like(new)
    )  # max(a-x, 0)
    """
    if state:
        if torch.any(original != new) and state.reg_vars:
            epsilon = 30

            cost += sum(
                [
                    10 * (1 - torch.exp(-((o - n) ** 2) * epsilon))
                    for o, n in zip(original, new)
                ]
            )
            # n = original.numel()
            # suma = (original - new).abs().sum() + n * epsilon
            # cost -= sum([((o - n).abs() + epsilon)/ suma * torch.log(((o - n).abs() + epsilon)/ suma) for o, n in zip(original, new)])

        if state.reg_int:
            # Calculate the distance with regularization
            # cost += (
            #     + (
            #         torch.tan(
            #             torch.pi
            #             * (new * state.metadata.dx_scaled + state.metadata.mean_scaled)
            #             * state.metadata.int_cols.to(device)
            #         )
            #     )
            #     ** 2
            # )
            cost += (
                + (
                        ((new * state.metadata.dx_scaled + state.metadata.mean_scaled) - 
                        torch.round(new * state.metadata.dx_scaled + state.metadata.mean_scaled))
                        * state.metadata.int_cols.to(device)
                    )
                ** 2
            )

    return (cost).sum() if with_sum else cost.sum(dim=1)


def unscale_instance(instance: torch.Tensor, metadata: DatasetMetadata, inplace: bool = False):
    """Undo scaling of a single instance."""

    cols_to_unscale = instance[metadata.cols_for_scaler].reshape(1, -1)
    mean = torch.tensor(metadata.scaler.mean_)
    std = torch.tensor(metadata.scaler.scale_)
    unscaled_cols = cols_to_unscale * std + mean
    if inplace:
        instance[metadata.cols_for_scaler] = torch.tensor(unscaled_cols, dtype=torch.float32).to(device)
        return instance
    else:
        instance_clone = instance.clone()
        instance_clone[metadata.cols_for_scaler] = torch.tensor(unscaled_cols, dtype=torch.float32).to(device)
        return instance_clone
    
def scale_instance(instance: torch.Tensor, metadata: DatasetMetadata, inplace: bool = False):
    """Scale a single instance according to the metadata scaler."""

    cols_to_scale = instance[metadata.cols_for_scaler].reshape(1, -1)
    mean = torch.tensor(metadata.scaler.mean_)
    std = torch.tensor(metadata.scaler.scale_)
    scaled_cols = (cols_to_scale - mean) / std
    if inplace:
        instance[metadata.cols_for_scaler] = torch.tensor(scaled_cols, dtype=torch.float32).to(device)
        return instance
    else:
        instance_clone = instance.clone()
        instance_clone[metadata.cols_for_scaler] = torch.tensor(scaled_cols, dtype=torch.float32).to(device)
        return instance_clone
    
def round_instance(instance: torch.Tensor, metadata: DatasetMetadata):
    """Round integer features of a scaled instance."""

    unscaled_person = unscale_instance(instance, metadata)
    unscaled_person[metadata.int_cols == 1] = torch.round(unscaled_person[metadata.int_cols == 1])
    person_new = scale_instance(unscaled_person, metadata)
    return person_new


def unscale_batch(batch: torch.Tensor, metadata: DatasetMetadata, inplace: bool = False):
    """Undo scaling for a batch of instances."""

    cols_to_unscale = torch.tensor(batch[:, metadata.cols_for_scaler], dtype=torch.float32)
    mean = torch.tensor(metadata.scaler.mean_, dtype=torch.float32)
    std = torch.tensor(metadata.scaler.scale_, dtype=torch.float32)
    unscaled_cols = cols_to_unscale * std + mean
    if inplace:
        batch[:, metadata.cols_for_scaler] = torch.tensor(unscaled_cols, dtype=torch.float32).to(device)
        return batch
    else:
        batch_clone = batch.clone()
        batch_clone[:, metadata.cols_for_scaler] = torch.tensor(unscaled_cols, dtype=torch.float32).to(device)
        return batch_clone
    
def scale_batch(batch: torch.Tensor, metadata: DatasetMetadata, inplace: bool = False):
    """Scale a batch of instances."""

    cols_to_scale = torch.tensor(batch[:, metadata.cols_for_scaler], dtype=torch.float32)
    mean = torch.tensor(metadata.scaler.mean_, dtype=torch.float32)
    std = torch.tensor(metadata.scaler.scale_, dtype=torch.float32)
    scaled_cols = (cols_to_scale - mean) / std
    if inplace:
        batch[:, metadata.cols_for_scaler] = torch.tensor(scaled_cols, dtype=torch.float32).to(device)
        return batch
    else:
        batch_clone = batch.clone()
        batch_clone[:, metadata.cols_for_scaler] = torch.tensor(scaled_cols, dtype=torch.float32).to(device)
        return batch_clone
    
def round_batch(batch: torch.Tensor, metadata: DatasetMetadata):
    """Round integer features in a batch of scaled instances."""

    unscaled_person = unscale_batch(batch, metadata)
    unscaled_person[metadata.int_cols == 1] = torch.round(unscaled_person[metadata.int_cols == 1])
    person_new = scale_batch(unscaled_person, metadata)
    return person_new


def newton_op(
    model: LogisticModel,
    person: torch.Tensor,
    metadata: DatasetMetadata,
    weights: torch.Tensor,
    delta_threshold: float = 0.1,
    max_epochs: int = 100,
    reg_int: bool = False,
    reg_vars: bool = False,
    reg_clamp: bool = False,
    print_: bool = False,
    der = False
):
    """Find a counterfactual instance using a Newton-like optimisation."""
    torch.manual_seed(0)
    output = model(person.unsqueeze(0))

    l = torch.nn.Parameter(torch.rand(1)[0]).to(device)
    person_new = person.clone().detach().to(device)

    state = 0

    if (
        torch.argmax(output, dim=1).item() == (1 - metadata.good_class)
    ):  # TODO: poner todo en función del threshold del modelo
        person_new.requires_grad = True

        lr = 1

        thres_term = (metadata.threshold - output[0][metadata.good_class]).item()

        state = State(
            model,
            metadata,
            max_epochs=max_epochs,
        )
        state.reg_vars = reg_vars
        state.reg_int = False
        state.reg_clamp = reg_clamp
        first_time = True
        continue_condition = True
        original_weights = weights.clone()

        while continue_condition and state.epochs < state.max_epochs:
            print("Epoch:", state.epochs) if print_ else None
            if (weights != 0).sum() == 1:
                ###################################################
                ### Si solo tenemos una variables activa que cambiar
                ###################################################
                derivative = torch.autograd.grad(
                    model(person_new.unsqueeze(0))[0][metadata.good_class],
                    person_new,
                    create_graph=True,
                    allow_unused=True,
                )[0][weights != 0]
                delta = torch.cat(
                    (
                        0.3 * (
                            model(person_new.unsqueeze(0))[0][metadata.good_class]
                            - state.metadata.threshold
                        )/ derivative,
                        torch.tensor([0]).to(device),
                    ),
                    dim=0,
                )
                # lr = 0.6
                # delta_threshold /= 4
            else:
                if (abs(thres_term) < 0.1) and first_time and state.epochs > 1:
                    ###################################################
                    ### Se estamos medio cerca de la solución, aplicamos reg_int y eliminamos reg_vars
                    ###################################################
                    if state.reg_vars:
                        change = (person - person_new).abs()
                        temp_weights = ((weights != 0) & (
                            change > 1e-2
                        )) * weights  # para mantener los numeros de weights
                        if state.reg_int:
                            cont_vars = (weights != 0) & ~metadata.int_cols.to(device)
                            changes_cont = cont_vars * change
                            temp_weights[torch.argmax(changes_cont)] = weights[
                                torch.argmax(changes_cont)
                            ] # Esto es por si
                        if temp_weights.sum() == 0:
                            # Si no hay cambios lo suficientemente grandes, dejamos libre dos variable,
                            temp_weights[torch.argmax(change)] = weights[
                                torch.argmax(change)
                            ]

                        weights = temp_weights
                        with torch.no_grad():
                            person_new[change <= 1e-2] = person[change <= 1e-2]
                        state.reg_vars = False

                    if reg_int:
                        state.reg_int = True

                    first_time = False
                # Reset the gradients
                person_new = person_new.detach().clone().requires_grad_(True)
                l = l.detach().clone().requires_grad_(True)

                ###################################################
                ### Definimos la función todas las veces para que tenga variables como statey weights que van cambiando
                ###################################################
                def fpl_func(x: torch.Tensor, l: torch.Tensor):
                    x = x.to(device)
                    out = model(x.unsqueeze(0))[0][metadata.good_class]
                    cost_func = torch.autograd.grad(
                        distance(person, x, weights, state),
                        x,
                        create_graph=True,
                        allow_unused=True,
                    )[0][weights != 0]
                    restriction = torch.autograd.grad(
                        out, x, create_graph=True, allow_unused=True
                    )[0][weights != 0]
                    l_derivative = (metadata.threshold - out).unsqueeze(0)
                    return torch.cat((cost_func - l * restriction, l_derivative))

                fpl = fpl_func(person_new, l)
                jac_tuple = torch.autograd.functional.jacobian(fpl_func, (person_new, l))

                if torch.linalg.norm(jac_tuple[1], ord=float("inf")) < delta_threshold:
                    ###################################################
                    ### Problemas con la hessiana
                    ###################################################
                    # ponderamos salto a como de lejos estemos
                    delta = (
                        thres_term
                        *
                        torch.cat(
                            (
                                jac_tuple[0][-1, weights != 0]
                                / torch.linalg.norm(jac_tuple[1]),
                                torch.tensor([0]).to(device),
                            ),
                            dim=0,
                        )
                    )
                    print(
                        f"ONLY MODEL DERIVATIVE: {torch.linalg.norm(jac_tuple[1], ord=float('inf'))}"
                    ) if print_ else None

                    

                else:
                    jac = torch.cat(
                        (jac_tuple[0][:, weights != 0], jac_tuple[1].unsqueeze(-1)),
                        dim=1,
                    )
                    delta = torch.linalg.inv(jac) @ fpl

            ###################################################
            ### Actualizamos con no_grad porque las operaciones inplace con autograd no funcionan
            ###################################################
            with torch.no_grad():
                person_new[weights != 0] -= delta[:-1] * lr
                l -= delta[-1] * lr
            # person_new = torch.clamp(person_new, metadata.min_values, metadata.max_values)


            ###################################################
            ### Clamping
            ###################################################
            if abs(thres_term) < 0.1 and state.epochs > 1 and state.reg_clamp:
                inbounds = torch.clamp(person_new, metadata.min_values.to(device), metadata.max_values.to(device)) == person_new
                # print(~inbounds) if inbounds.all() == False else None
                # weights = ((original_weights != 0) & inbounds) * original_weights
                weights = ((weights != 0) & inbounds) * weights
                if weights.sum() == 0:
                    # Si no hay cambios lo suficientemente grandes, dejamos libre dos variable,
                    weights[metadata.int_cols] = original_weights[metadata.int_cols]
                    
                person_new = torch.clamp(
                    person_new,
                    metadata.min_values.to(device),
                    metadata.max_values.to(device),
                )

            output_new = model(person_new.unsqueeze(0))
            thres_term = (metadata.threshold - output_new[0][metadata.good_class]).item()
            state.epochs += 1

            if print_:
                print(
                    "Changes:",
                    " delta1:",
                    delta[0].item(),
                    " delta_l:",
                    delta[-1].item(),
                )
                print(
                    "dist:",
                    distance(person, person_new, weights, state).item(),
                    ", threshold:",
                    thres_term,
                )

            ###################################################
            ### Actualizamos la condición de salida
            ###################################################
            # abs(thres_term) > 1e-15 or 
            continue_condition = (thres_term > 0) or torch.linalg.norm(
                delta
            ) > 1e-6 * torch.linalg.norm(torch.cat((person_new, l.unsqueeze(0))))
            if not continue_condition and state.reg_int:
                ###################################################
                ### Si ya salimos pero estabamos aplicando reg_int,
                ### entonces aplicamos el redondeo y volvemos a optimizar
                ###################################################
                weights = ((weights != 0) & ~metadata.int_cols.to(device)) * weights
                # if temp_weights.sum() == 0:

                person_new = round_instance(person_new, metadata)
                state.reg_int = False
                continue_condition = True

        if print_:
            print("Original output:", model(person.unsqueeze(0)))
            print("New output:", output_new)
            # print("Original input unscaled:", unscale_instance(person, metadata))
            # print("New input unscaled:", unscale_instance(person_new, metadata))
            print("Regularization strength:", l.item())
            print("Epochs:", state.epochs)
    return person_new, state

if __name__ == "__main__":
    filename = "data/adult_income_train.csv"
    model_name = "model_adult_income_train"

    person: torch.Tensor
    metadata: DatasetMetadata
    person, metadata = load_data(filename, index=0)

    # define model
    model: LogisticModel = main_train(filename, model_name)
    # model: LogisticModel = load_model(model_name).to(device)

    outputs = model(person).argmax(dim=1)
    if outputs == 0:
        print("Person is not a defaulter")
        exit(0)

    weights = torch.tensor(metadata.cols_for_mask, dtype=torch.int).to(device)

    
