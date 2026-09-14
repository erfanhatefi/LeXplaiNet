import torch
from .rule_utils import (
    apply_linear_operation,
    preserve_forward_with_modified_gradient,
    _positive_bias,
    _negative_bias,
)


def epsilon_rule(self, x, ignore_bias=False):
    """
    Implicit implementation of LRP-Epsilon(zero) rule for Linear and Conv2d layers.
    Given the higher level relevances computed by GxI framework, the implicit LRP-Epsilon zero
    is also the same Gradient*Input framework. We do not need the epsilon here anymore as there
    is no division operation taking place in the backward pass.

    Check this paper for further information: Arras, Leila, et al. "A close look at decomposition-based XAI-methods
    for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

    It is suggested (but have to check it on your own) to not ignore (flag=False) of lieanr layers
    everywhere, and ignore the conv2d (flag=True)
    Reason probably: Linear layers, especially after LayerNorm or near the classifier, can have meaningful bias terms.

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.

    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    w = self.weight

    if ignore_bias:
        original_output = self.old_forward(x)
        z_proxy = apply_linear_operation(self, x, w, bias=None)
        return preserve_forward_with_modified_gradient(original_output, z_proxy)

    return apply_linear_operation(self, x, w)


def zplus_rule(self, x, ignore_bias=True):
    """
    Implicit implementation of LRP-ZPlus (or Alpha1-Beta0) rule for Linear and Conv2d layers.
    This can be computed in implicit way using $\hat z_j = z_j^{pos} [\frac{z_j}{z_j^{pos}}]_\texttt{.detach()}$
    such that $z_j^{pos} = \sum_i (x_i w_{ij})^+ + b_j^+$.

    Check this paper for further information: Arras, Leila, et al. "A close look at decomposition-based XAI-methods
    for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.
    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    w = self.weight
    pos_w, neg_w = torch.clamp(w, min=0), torch.clamp(w, max=0)

    pos_bias = None if ignore_bias else _positive_bias(self)
    neg_bias = None if ignore_bias else _negative_bias(self)

    def z_pos(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=pos_bias)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=neg_bias)
        return vp + vn

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_pos(x))

    return z_hat


def alphabeta_rule(self, x, alpha, beta, ignore_bias=True):
    """
    Implicit implementation of LRP-AlphaBeta rule for Linear and Conv2d layers.
    This can be computed in implicit way using $\hat z_j = \alpha \cdot z_j^{pos}
      [\frac{z_j}{z_j^{pos}}]_\texttt{.detach()} + \beta \cdot z_j^{neg}
     [\frac{z_j}{z_j^{neg}}]_\texttt{.detach()}$ such that $z_j^{pos} = \sum_i
       (x_i w_{ij})^+ + b_j^+$ and $z_j^{neg} = \sum_i (x_i w_{ij})^- + b_j^-$. We
       are the first to show this formulation!

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.
        alpha (float): The alpha parameter for the rule.
        beta (float): The beta parameter for the rule.
    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    assert alpha + beta == 1, "alpha - beta must be equal to 1"

    w = self.weight
    pos_w, neg_w = torch.clamp(w, min=0), torch.clamp(w, max=0)

    pos_bias = None if ignore_bias else _positive_bias(self)
    neg_bias = None if ignore_bias else _negative_bias(self)

    def z_pos(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=pos_bias)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=neg_bias)
        return vp + vn

    def z_neg(x):
        vp = apply_linear_operation(self, torch.clamp(x, max=0), pos_w, bias=pos_bias)
        vn = apply_linear_operation(self, torch.clamp(x, min=0), neg_w, bias=neg_bias)
        return vp + vn

    z_hat = alpha * preserve_forward_with_modified_gradient(
        self.old_forward(x), z_pos(x)
    ) + beta * preserve_forward_with_modified_gradient(self.old_forward(x), z_neg(x))
    return z_hat


def gamma_rule(self, x, gamma, ignore_bias=True):
    """
    Implicit implementation of LRP-Gamma rule for Linear and Conv2d layers.
    This can be computed in implicit way using $\hat z_j = \tilde z_j
    [\frac{z_j}{\tilde z_j}]_\texttt{.detach()}$ such that $\tilde z_j =
    \sum_i x_i w_{ij} + \gamma (x_i w_{ij})^+ + b_j$. This formulation
    has been shown here for the first time!

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.
        gamma (float): The gamma parameter for the rule.
    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    w = self.weight
    pos_w, neg_w = torch.clamp(w, min=0), torch.clamp(w, max=0)

    bias = None if ignore_bias else self.bias
    pos_bias = None if ignore_bias else _positive_bias(self)
    neg_bias = None if ignore_bias else _negative_bias(self)

    def z_tilde(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=pos_bias)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=neg_bias)
        return apply_linear_operation(self, x, w, bias=bias) + (vp + vn) * gamma

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_tilde(x))
    return z_hat


def inverse_gamma_rule(self, x, gamma, ignore_bias=True):
    """
    Implicit implementation of LRP-Gamma rule for Linear and Conv2d layers.
    It is the same as the "gamma_rule" but with the gamma parameter being
      applied in the opposite way.
    This can be computed in implicit way using $\hat z_j = \tilde z_j
    [\frac{z_j}{\tilde z_j}]_\texttt{.detach()}$ such that $\tilde z_j =
    \sum_i x_i w_{ij} + \gamma (x_i w_{ij})^+ + b_j$. This formulation
    has been shown here for the first time!

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.
        gamma (float): The gamma parameter for the rule.
    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    w = self.weight
    pos_w, neg_w = torch.clamp(w, min=0), torch.clamp(w, max=0)

    bias = None if ignore_bias else self.bias
    pos_bias = None if ignore_bias else _positive_bias(self)
    neg_bias = None if ignore_bias else _negative_bias(self)

    def z_tilde(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=pos_bias)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=neg_bias)
        return apply_linear_operation(self, x, w, bias=bias) / gamma + (vp + vn)

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_tilde(x))
    return z_hat


def lifted_gamma_rule(self, x, gamma, ignore_bias=True):
    """
    Lifted gamma is an upgraded version of the gamma rule, where we lift the gamma
    parameter to a higher value while ensuring that the relevance scores do not
    explode. This can be computed in implicit way using the original formulation
    of the gamma rule but with the lifted gamma value. The lifted gamma value
    can be computed as $\hat \gamma = \min(1.0 - (\frac{z_j}{z_j^{pos}})^2, \gamma)$,
    where $z_j^{pos} = \sum_i (x_i w_{ij})^+ + b_j^+$.

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.
        gamma (float): The gamma parameter for the rule.
    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    w = self.weight
    pos_w, neg_w = torch.clamp(w, min=0), torch.clamp(w, max=0)

    bias = None if ignore_bias else self.bias
    pos_bias = None if ignore_bias else _positive_bias(self)
    neg_bias = None if ignore_bias else _negative_bias(self)

    def z_tilde(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=pos_bias)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=neg_bias)
        pos_contr = vp + vn
        u = apply_linear_operation(self, x, w, bias=bias)

        epsthresh = 1e-10
        pstab = torch.where(
            pos_contr > epsthresh, pos_contr, torch.full_like(pos_contr, epsthresh)
        )
        min_gamma = (1.0 - u / pstab) ** 2
        gamma_eff = torch.maximum(
            torch.as_tensor(gamma, device=x.device, dtype=x.dtype), min_gamma
        )

        return pos_contr + u / gamma_eff

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_tilde(x))
    return z_hat


class DivideGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / ctx.factor, None


def uniform_gradient_division_rule(x, detached_factor=2):
    """
    This is a simple rule that uniformly divides the gradient by a factor of "detached_factor"
    and detaches it from the computational graph. It can be used based on your choice. E.g., in
    some cases you might be interested to uniformly divide the relevance scores between the matrices
    in a matrix to matrix multiplication operation, which can be easily done using this rule.

    Check this paper for further information: Arras, Leila, et al. "A close look at decomposition-based XAI-methods
    for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

    Args:
        x (torch.Tensor): The input tensor to which the rule is applied.
        detached_factor (float): The factor by which the gradient is uniformly divided and detached.
    Returns:
        torch.Tensor: The output tensor resulting from applying the uniform gradient division rule.
    """
    # fraction = 1 / detached_factor
    # z_hat = x * fraction + (x * (1 - fraction)).detach()
    # return z_hat
    # Alternative Approach which is more efficient and does not
    # require any additional memory allocation for the detached tensor.
    # the main benefit is in the numerical error causes by multiplying
    # and adding the detached tensor, which can be avoided by using this approach.
    return DivideGradient.apply(x, detached_factor)


def block_rule(self, x):
    """
    This rule is similar to detaching a block of the computational graph.
    It can be used to block the relevance flow through a layer. We use this
    to have a control over of the relevance flow and modules inside the model.

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.
    Returns:
        torch.Tensor: The output tensor resulting from applying the block rule.
    """
    z_hat = self.old_forward(x).detach()
    return z_hat


def identity_rule(self, x):
    """
    Identity rule ignores the recomputaion of relevance or gradient over a specific layer
    (mainly the element-wise activation functions) and just passes the relevance scores or
    gradients through it without any change.

    Check this paper for further information: Arras, Leila, et al. "A close look at decomposition-based XAI-methods
    for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to which the identity rule is applied.
    Returns:
        torch.Tensor: The output tensor resulting from applying the identity rule.
    """
    return preserve_forward_with_modified_gradient(self.old_forward(x), x)


def dropout_rule(self, x):
    """
    We normally ignore the dropout layers in the relevance flow, as they are not present
    during inference and we want to have a control over the relevance flow. This rule can
    be used to ignore the dropout layers in the relevance flow.

    Args:
        self (torch.nn.Module): The dropout layer on which the rule is applied.
        x (torch.Tensor): The input tensor to the dropout layer.
    Returns:
        torch.Tensor: The output tensor resulting from applying the dropout rule.
    """
    # here we ignore dropout
    return x
