import torch


def stabilize(x: torch.Tensor, eps: float = 1e-6):
    """
    Stabilize the input by adding a small epsilon value to prevent prospective division
      by zero or very small numbers.

    Args:
        x (torch.Tensor): The input tensor to be stabilized.
        eps (float, optional): A small constant value to add to the input for stabilization.
                               Default is 1e-6.
    Returns:
        torch.Tensor: The stabilized input tensor.
    """
    return x + eps


def apply_operation(self, x: torch.Tensor, w: torch.Tensor):
    """
    Apply the linear or convolutional operation using the provided weights and the input tensor.
    This function checks the type of the layer (Linear or Conv2d) and applies the corresponding
    operation using the provided weights. It also handles the presence of bias in the layer.

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the operation is to be applied.
        x (torch.Tensor): The input tensor to the layer.
        w (torch.Tensor): The weights to be used for the operation.
    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    if isinstance(self, torch.nn.Linear):
        # check bias
        if self.bias is not None:
            return torch.nn.functional.linear(x, w, self.bias)
        else:
            return torch.nn.functional.linear(x, w)
    elif isinstance(self, torch.nn.Conv2d):
        # check bias
        if self.bias is not None:
            return torch.nn.functional.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            return torch.nn.functional.conv2d(
                x, w, None, self.stride, self.padding, self.dilation, self.groups
            )
    else:
        raise NotImplementedError(
            "apply_operation only supports Linear and Conv2d layers."
        )
