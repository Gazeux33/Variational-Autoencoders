import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float]:
    model.train()

    total_loss = 0
    total_recons_loss = 0
    total_kl_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        z_mean, z_log_var, reconstruction = model(X)
        total_loss_val, recon_loss, kld_loss = loss_fn(X, reconstruction, z_mean, z_log_var)

        total_loss += total_loss_val.item()
        total_recons_loss += recon_loss.item()
        total_kl_loss += kld_loss.item()

        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

    total_loss /= len(dataloader)
    total_recons_loss /= len(dataloader)
    total_kl_loss /= len(dataloader)
    return total_loss, total_recons_loss, total_kl_loss


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0
    total_recons_loss = 0
    total_kl_loss = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            z_mean, z_log_var, reconstruction = model(X)
            total_loss_val, recon_loss, kld_loss = loss_fn(X, reconstruction, z_mean, z_log_var)

            total_loss += total_loss_val.item()
            total_recons_loss += recon_loss.item()
            total_kl_loss += kld_loss.item()

    total_loss /= len(dataloader)
    total_recons_loss /= len(dataloader)
    total_kl_loss /= len(dataloader)
    return total_loss, total_recons_loss, total_kl_loss


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    results = {"train_total": [],
               "train_recons": [],
               "train_kl": [],
               "test_total": [],
               "test_recons": [],
               "test_kl": []
               }
    for epoch in tqdm(range(epochs)):
        train_total, train_recons, train_kl = train_step(model=model,
                                                         dataloader=train_dataloader,
                                                         loss_fn=loss_fn,
                                                         optimizer=optimizer,
                                                         device=device)
        test_total, test_recons, test_kl = test_step(model=model,
                                                     dataloader=test_dataloader,
                                                     loss_fn=loss_fn,
                                                     device=device)
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_total:.4f} | "
            f"test_loss: {test_total:.4f} | "
        )
        results["train_total"].append(train_total)
        results["train_recons"].append(train_recons)
        results["train_kl"].append(train_kl)

        results["test_total"].append(test_total)
        results["test_recons"].append(test_recons)
        results["test_kl"].append(test_kl)

    return results
