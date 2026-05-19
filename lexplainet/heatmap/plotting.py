import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .zennit_image import imgify
from torchvision import transforms


def generate_heatmap(
    relevance,
):
    # check dimension:
    if len(relevance.shape) == 4:
        relevance = relevance.sum(dim=(0, 1))  # sum over channels
    elif len(relevance.shape) == 3:
        relevance = relevance.sum(dim=0)  # sum over channels
    # check if relevance is normalized
    if relevance.min() < -1 or relevance.max() > 1:
        print("Warning: Relevance scores are not normalized between -1 and 1.")
        relevance = relevance / torch.max(torch.abs(relevance))

    heatmap = imgify(
        relevance.squeeze(0).cpu(),
        vmin=relevance.min().cpu(),
        vmax=relevance.max().cpu(),
        cmap="bwr",
    )
    return heatmap


def plot_heatmap_zennit(relevance):
    heatmap = generate_heatmap(relevance)
    # make a figue with matplotlib
    plt.figure(figsize=(6, 6))
    # put image in middle without axis and white space borders
    plt.imshow(heatmap)
    plt.axis("off")
    plt.show()
    plt.tight_layout()


def plot_heatmap(relevance, q=100, show=True, save=False, save_path="heatmap.png"):
    # if relevance has extra dimension, remove it
    if len(relevance.shape) == 4:
        heatmap = relevance.sum(dim=(0, 1))  # sum over channels
    elif len(relevance.shape) == 3 and relevance.shape[0] == 1:
        heatmap = relevance.sum(dim=0)  # sum over channels

    heatmap = heatmap.cpu().numpy()

    clim = np.percentile(np.abs(heatmap), q)

    heatmap = heatmap / clim

    if save:
        plt.imsave(save_path, heatmap, cmap="seismic", vmin=-1, vmax=1)
    if show == False:
        return heatmap, clim
    else:
        show_heatmap(heatmap, clim)


def show_heatmap(heatmap, clim):
    plt.imshow(heatmap, cmap="seismic", clim=(-clim, clim))
    plt.axis("off")


def load_image(image_path, transform=None, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    if transform is not None:
        # to tensor and add batch dimension
        image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device) if transform is not None else image


def show_image(source, transform=None):
    if type(source) == str:
        # load image from path show it via matplotlib
        image = load_image(source, transform=transform)
        show_image(image)
    elif type(source) == torch.Tensor:
        print("Image Tensor Shape:", source.shape)
        unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image = unnormalize(source.squeeze(0).cpu())
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        plt.tight_layout()
