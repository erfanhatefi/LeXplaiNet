import torch
from .rule_utils import (
    apply_linear_operation,
    preserve_forward_with_modified_gradient,
    _positive_bias,
    _negative_bias,
)


def epsilon_rule(self, x):
    """
    Implicit implementation of LRP-Epsilon(zero) rule for Linear and Conv2d layers.
    Given the higher level relevances computed by GxI framework, the implicit LRP-Epsilon zero
    is also the same Gradient*Input framework. We do not need the epsilon here anymore as there
    is no division operation taking place in the backward pass.

    Check this paper for further information: Arras, Leila, et al. "A close look at decomposition-based XAI-methods
    for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

    Args:
        self (torch.nn.Module): The layer (Linear or Conv2d) on which the
        x (torch.Tensor): The input tensor to the layer.

    Returns:
        torch.Tensor: The output tensor resulting from applying the operation with the provided weights.
    """
    w = self.weight
    z = apply_linear_operation(self, x, w)
    return z


def zplus_rule(self, x):
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

    def z_pos(x):
        vp = apply_linear_operation(
            self, torch.clamp(x, min=0), pos_w, bias=_positive_bias(self)
        )
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=None)
        return vp + vn

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_pos(x))

    return z_hat


def alphabeta_rule(self, x, alpha, beta):
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
    w = self.weight
    pos_w, neg_w = torch.clamp(w, min=0), torch.clamp(w, max=0)

    def z_pos(x):
        vp = apply_linear_operation(
            self, torch.clamp(x, min=0), pos_w, bias=_positive_bias(self)
        )
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=None)
        return vp + vn

    def z_neg(x):
        vp = apply_linear_operation(
            self, torch.clamp(x, max=0), pos_w, bias=_negative_bias(self)
        )
        vn = apply_linear_operation(self, torch.clamp(x, min=0), neg_w, bias=None)
        return vp + vn

    z_hat = alpha * preserve_forward_with_modified_gradient(
        self.old_forward(x), z_pos(x)
    ) + beta * preserve_forward_with_modified_gradient(self.old_forward(x), z_neg(x))
    return z_hat


def gamma_rule(self, x, gamma):
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

    def z_tilde(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=None)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=None)
        return apply_linear_operation(self, x, w) + (vp + vn) * gamma

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_tilde(x))
    return z_hat


def inverse_gamma_rule(self, x, gamma):
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

    def z_tilde(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=None)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=None)
        return apply_linear_operation(self, x, w) / gamma + (vp + vn)

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_tilde(x))
    return z_hat


def lifted_gamma_rule(self, x, gamma):
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

    def z_tilde(x):
        vp = apply_linear_operation(self, torch.clamp(x, min=0), pos_w, bias=None)
        vn = apply_linear_operation(self, torch.clamp(x, max=0), neg_w, bias=None)
        positive_contribution = vp + vn
        u = apply_linear_operation(self, x, w)

        eps_threshold = 1e-10
        vpstab = torch.where(
            positive_contribution > eps_threshold, positive_contribution, eps_threshold
        )
        mingamma = (1.0 - u / vpstab) ** 2
        lifted_gamma = torch.clamp(mingamma, max=gamma)

        return u + (positive_contribution * lifted_gamma)

    z_hat = preserve_forward_with_modified_gradient(self.old_forward(x), z_tilde(x))
    return z_hat


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
    fraction = 1 / detached_factor
    z_hat = x * fraction + (x * (1 - fraction)).detach()
    return z_hat


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
