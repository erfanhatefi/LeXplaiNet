import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .zennit_image import imgify
from torchvision import transforms


def _to_numpy_image(image):
    if image is None:
        return None

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image.squeeze(0)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image_min = image.min()
    image_max = image.max()
    if image_min < 0.0 or image_max > 1.0:
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        elif image_max > 1.0:
            image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _resize_image(image, size):
    h, w = size
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    pil_image = pil_image.resize((w, h), Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.float32) / 255.0


def _convolve2d(image, kernel):
    padded = np.pad(image, 1, mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            output += (
                kernel[y, x] * padded[y : y + image.shape[0], x : x + image.shape[1]]
            )
    return output


def generate_edge_mask(image, shape=None, threshold=0.75):
    """Generate a cheap Sobel edge mask from an image.

    Parameters
    ----------
    image : str, PIL.Image, torch.Tensor, or np.ndarray
        Original input image.
    shape : tuple, optional
        Target ``(height, width)`` for the edge mask.
    threshold : float
        Percentile-like threshold in [0, 1]. Higher values keep fewer edges.
    """

    image = _to_numpy_image(image)
    if image is None:
        return None
    if shape is not None and image.shape[:2] != tuple(shape):
        image = _resize_image(image, shape)

    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    edges = np.hypot(_convolve2d(gray, sobel_x), _convolve2d(gray, sobel_y))
    max_edge = edges.max()
    if max_edge > 0:
        edges = edges / max_edge

    cutoff = np.quantile(edges, np.clip(threshold, 0.0, 1.0))
    edges = np.where(edges >= cutoff, edges, 0.0)
    max_edge = edges.max()
    if max_edge > 0:
        edges = edges / max_edge
    return edges


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
    fig, ax = make_heatmap_figure(heatmap, cmap=None)
    plt.show()
    return fig, ax


def prepare_heatmap(relevance, q=100):
    """Convert a relevance tensor to a clipped, normalized 2D heatmap."""

    if len(relevance.shape) == 4:
        heatmap = relevance.sum(dim=(0, 1))  # sum over channels
    elif len(relevance.shape) == 3 and relevance.shape[0] == 1:
        heatmap = relevance.sum(dim=0)  # sum over channels
    elif len(relevance.shape) == 3:
        heatmap = relevance.sum(dim=0)
    else:
        heatmap = relevance

    heatmap = heatmap.cpu().numpy()
    clim = np.percentile(np.abs(heatmap), q)
    if clim == 0:
        clim = np.max(np.abs(heatmap))
    if clim == 0:
        clim = 1.0

    heatmap = np.clip(heatmap / clim, -1, 1)
    return heatmap, clim


def make_heatmap_figure(
    heatmap,
    cmap="seismic",
    image=None,
    edge_overlay=False,
    edge_threshold=0.75,
    edge_alpha=0.45,
    edge_color="black",
    dpi=100,
):
    h, w = heatmap.shape[:2]
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_alpha(0)
    draw_heatmap_on_axis(
        ax,
        heatmap,
        cmap=cmap,
        image=image,
        edge_overlay=edge_overlay,
        edge_threshold=edge_threshold,
        edge_alpha=edge_alpha,
        edge_color=edge_color,
    )
    ax.set_position([0, 0, 1, 1])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def draw_heatmap_on_axis(
    ax,
    heatmap,
    cmap="seismic",
    image=None,
    edge_overlay=False,
    edge_threshold=0.75,
    edge_alpha=0.45,
    edge_color="black",
):
    """Draw a prepared heatmap on an existing Matplotlib axis."""

    if heatmap.ndim == 2:
        ax.imshow(heatmap, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    else:
        ax.imshow(heatmap, interpolation="nearest")

    if edge_overlay:
        edges = generate_edge_mask(
            image, shape=heatmap.shape[:2], threshold=edge_threshold
        )
        if edges is None:
            raise ValueError(
                "edge_overlay=True requires the original image via image=..."
            )
        edge_rgb = np.zeros((*edges.shape, 4), dtype=np.float32)
        if edge_color == "white":
            edge_rgb[..., :3] = 1.0
        elif edge_color == "gray":
            edge_rgb[..., :3] = 0.35
        else:
            edge_rgb[..., :3] = 0.0
        edge_rgb[..., 3] = np.clip(edges * edge_alpha, 0.0, 1.0)
        ax.imshow(edge_rgb, interpolation="nearest")

    ax.set_axis_off()
    return ax


def plot_heatmap(
    relevance,
    q=100,
    show=True,
    save=False,
    save_path="heatmap.png",
    image=None,
    edge_overlay=False,
    edge_threshold=0.9,
    edge_alpha=0.2,
    edge_color="gray",
    cmap="seismic",
    dpi=100,
):

    # check if image is detached and on cpu or not
    if isinstance(image, torch.Tensor):
        if image.requires_grad:
            image = image.detach()
        if image.is_cuda:
            image = image.cpu()

    heatmap, clim = prepare_heatmap(relevance, q=q)

    fig, ax = make_heatmap_figure(
        heatmap,
        cmap=cmap,
        image=image,
        edge_overlay=edge_overlay,
        edge_threshold=edge_threshold,
        edge_alpha=edge_alpha,
        edge_color=edge_color,
        dpi=dpi,
    )

    if save:
        fig.savefig(save_path, dpi=dpi, pad_inches=0, transparent=True)
    if show == False:
        plt.close(fig)
        return heatmap, clim
    else:
        plt.show()


def show_heatmap(heatmap, clim=None, **kwargs):
    # old version
    # plt.imshow(heatmap, cmap="seismic", clim=(-clim, clim))
    # plt.axis("off")
    fig, ax = make_heatmap_figure(heatmap, **kwargs)
    plt.show()
    return fig, ax


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
