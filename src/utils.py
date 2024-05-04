import torch
from pathlib import Path
import matplotlib.pyplot as plt

from datetime import datetime
import os
from typing import Dict
import json


def sample_from_latent_space(model, zdim, num):
    z = torch.randn(size=(num, zdim))
    with torch.no_grad():
        decode = model(z)
    return decode


def sample_from_latent_space2(model, zdim, num, dataloader):
    with torch.no_grad():
        examples = next(iter(dataloader))
        inputs = examples[0]
        z_mean, z_log_var = model.encoder(inputs)
        z = model.reparameterize(z_mean, z_log_var)
    zmax = z.max(dim=1)[0].max()
    zmin = z.min(dim=1)[0].min()
    random_tensor = torch.rand(num, zdim) * (zmax - zmin) + zmin
    with torch.no_grad():
        decode = model.decoder(random_tensor).squeeze()
    return random_tensor, decode


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
        z_mean, z_log_var = model.encoder(inputs)
        z = model.reparameterize(z_mean, z_log_var)
        print(z.shape)
        reconstructions = model.decoder(z)
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


def plot_results(results: dict, save_path: str = "plots"):
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for key in results:
        plt.plot(results[key], label=key)
    plt.legend()
    plt.grid()

    # Save the plot with a unique filename using current date and time
    save_file = os.path.join(save_path, f"plot_{current_time}.png")
    plt.savefig(save_file)

    plt.show()


def create_image_gallery(images, images_per_row=10):
    num_images = images.shape[0]

    num_rows = (num_images + images_per_row - 1) // images_per_row

    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 1.5, num_rows * 1.5))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray', interpolation='nearest')
        ax.axis('off')

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
