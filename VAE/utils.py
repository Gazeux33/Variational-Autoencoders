import torch
from pathlib import Path
import matplotlib.pyplot as plt

from typing import Dict
import json


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def predict(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader):
    with torch.no_grad():
        examples = next(iter(dataloader))
        inputs = examples[0]
        _, _, reconstructions = model(inputs)
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(inputs[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructions[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()


def save_results(obj: Dict,
                 path: str):
    with open(path, "w+") as file:
        json.dump(obj, file)
