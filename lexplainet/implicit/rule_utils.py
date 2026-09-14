import torch

USE_MODULE_BIAS = object()


def _positive_bias(module):
    return None if module.bias is None else torch.clamp(module.bias, min=0)


def _negative_bias(module):
    return None if module.bias is None else torch.clamp(module.bias, max=0)


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


class PreserveForwardWithModifiedGradient(torch.autograd.Function):
    """
    A custom autograd function to preserve the forward output while modifying the gradient.
    """

    @staticmethod
    def forward(ctx, original_output, gradient_proxy):
        ctx.save_for_backward(original_output, gradient_proxy)
        return original_output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        original_output, gradient_proxy = ctx.saved_tensors

        nonzero = gradient_proxy != 0
        safe_proxy = torch.where(
            nonzero,
            gradient_proxy,
            torch.ones_like(gradient_proxy),
        )

        scale = torch.where(
            nonzero,
            original_output / safe_proxy,
            torch.zeros_like(gradient_proxy),
        )

        return None, grad_output * scale


def preserve_forward_with_modified_gradient(original_output: torch.Tensor, gradient_proxy: torch.Tensor):
    """
    Return ``gradient_proxy * (original_output / gradient_proxy).detach()`` while
    preserving the forward value when ``gradient_proxy`` is zero as it might
    be causing numerical instability due to division by zero. Instead of
    having a 0 * (value / 0).detach() which would be NaN, we want to have
      0 * (1).detach() which is 0, thus preserving the forward value of ``original_output``.

    This keeps forward values equal to ``original_output`` while steering gradients
    through ``gradient_proxy`` for implicit LRP.

    The new recent change uses PreserveForwardWithModifiedGradient autograd function to handle
    the backward pass correctly, ensuring that the gradients are computed as intended while
    preserving the forward output. The previous implementation had some issues with forward
    pass preservation due to the numerical errors that could arise from division by zero or
    very small numbers. The new implementation ensures that the forward output is preserved
    and passed exactly correctly while allowing gradients to flow through the gradient proxy.
    The previous version also passed everything correcetly and mathematically, however, it was
    not as numerically stable as the new implementation.
    """
    #     # create a boolean mask tensor, same shape as gradient_proxy
    #     # e.g. gradient_proxy = tensor([2.0, 0.0, -3.0])
    #     # nonzero        = tensor([True, False, True])
    #     nonzero = gradient_proxy != 0
    #     safe_gradient_proxy = torch.where(
    #         nonzero, gradient_proxy, torch.ones_like(gradient_proxy)
    #     )

    #     modified_output = gradient_proxy * (original_output / safe_gradient_proxy).detach()

    #     # where gradient_proxy is nonzero:
    #     #     use rescaled proxy, because it preserves forward and gives proxy gradient
    #     # where gradient_proxy is zero:
    #     #     use original_output.detach(), because rescaled would incorrectly become zero
    #     # An Example:
    #     # original_output      = tensor([10.0, 5.0, 6.0])
    #     # gradient_proxy = tensor([2.0, 0.0, -3.0])

    #     # nonzero        = tensor([True, False, True])
    #     # safe_gradient_proxy  = tensor([2.0, 1.0, -3.0])

    #     # ratio          = tensor([5.0, 5.0, -2.0]).detach()
    #     # modified_output       = gradient_proxy * ratio
    #     #             = tensor([10.0, 0.0, 6.0])
    #     # Thus:
    #     return torch.where(nonzero, modified_output, original_output.detach())
    return PreserveForwardWithModifiedGradient.apply(original_output, gradient_proxy)


def apply_linear_operation(
    self, x: torch.Tensor, w: torch.Tensor, bias=USE_MODULE_BIAS
):
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
    if bias is USE_MODULE_BIAS:
        bias = self.bias

    if isinstance(self, torch.nn.Linear):
        # check bias
        if bias is not None:
            return torch.nn.functional.linear(x, w, bias)
        else:
            return torch.nn.functional.linear(x, w)
    elif isinstance(self, torch.nn.Conv2d):
        # check bias
        if bias is not None:
            return torch.nn.functional.conv2d(
                x, w, bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            return torch.nn.functional.conv2d(
                x, w, None, self.stride, self.padding, self.dilation, self.groups
            )
    else:
        raise NotImplementedError(
            "apply_operation only supports Linear and Conv2d layers."
        )
