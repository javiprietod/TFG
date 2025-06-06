# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    correct: int = 0
    elements: int = 0
    losses: list[float] = []
    model.train()

    for user, target in train_data:
        user = user.to(device).float()
        target = target.to(device).long()

        optimizer.zero_grad()

        outputs = model(user)
        loss_value = loss(outputs.float(), target)

        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        elements += target.shape[0]
        outputs = torch.argmax(outputs, dim=1)
        correct += torch.sum(outputs == target).item()

    print(f"Epoch {epoch} - train accuracy: {correct / elements}")

    # write on tensorboard
    writer.add_scalar("train/accuracy", correct / elements, epoch)
    writer.add_scalar("train/loss", np.mean(losses), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """
    model.eval()
    with torch.no_grad():
        # define metric lists
        correct: int = 0
        elements: int = 0
        losses: list[float] = []

        for user, target in val_data:
            user = user.to(device).float()
            target = target.to(device).long()

            outputs = model(user)
            loss_value = loss(outputs.float(), target)
            losses.append(loss_value.item())
            elements += target.shape[0]
            outputs = torch.argmax(outputs, dim=1)
            correct += torch.sum(outputs == target).item()

        # write on tensorboard
        print(f"Epoch {epoch} - val accuracy: {correct / elements}")
        writer.add_scalar("val/accuracy", correct / elements, epoch)
        writer.add_scalar("val/loss", np.mean(losses), epoch)


def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """
    with torch.no_grad():
        correct: int = 0
        elements: int = 0
        all_preds = []
        all_targets = []
        model.eval()

        for user, target in test_data:
            user = user.to(device).float()
            target = target.to(device)

            outputs = model(user)

            # For binary classification: assuming the outputs are logits/scores, we take the max index for prediction
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            elements += target.shape[0]
            correct += (preds == target).sum().item()

        # Calculate metrics
        accuracy = correct / elements
        f1 = f1_score(
            all_targets, all_preds, average="macro"
        )  # You can change to 'macro' or 'micro' if needed
        conf_matrix = confusion_matrix(all_targets, all_preds)

        return accuracy, f1, conf_matrix
