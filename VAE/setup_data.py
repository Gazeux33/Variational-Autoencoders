import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
        root: str,
        transform: transforms.Compose,
        batch_size: int,
):
    train_data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloader, test_dataloader
