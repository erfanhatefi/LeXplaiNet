import torch
from .core import check_already_patched
from .rules import uniform_gradient_division_rule

"""
This module contains the model-specific patches for the LRP implementation.
These patches are designed to handle specific layers or operations in certain
models that require special treatment for LRP. For example, in transformer-based models,
we need to patch the attention mechanism to apply the uniform rule in matmul operations
via the Gradient*Input framework. Similarly, for ResNet models, we need to canonize the
Conv2d and BatchNorm2d layers to ensure that the relevance scores are correctly propagated
through these layers. The functions in this module are responsible for applying these patches
to the respective layers or operations in the models.
"""


##########################################
########### For Transformers #############
##########################################


def patch_attention(module: torch.nn.Module):
    """
    This function has been adapted from https://github.com/rachtibat/LRP-eXplains-Transformers
    Huggingface's transformers library provides a dictionary of all attention functions.
    We patch all of them with the same wrapper function to implement the uniform rule in
    matmul operations via the Gradient*Input framework. It is sufficient to correct the
    gradient flow later at the query, key, and value tensors.

    Args:
        module (torch.nn.Module): The attention module whose forward method is to be patched.
    Returns:
        bool: True if the patching was successful, False if the module was already patched.
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    else:
        module.eager_attention_forward = new_forward

    NEW_ATTENTION_FUNCTIONS = {}
    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        new_forward = wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        else:
            NEW_ATTENTION_FUNCTIONS[key] = new_forward
    module.ALL_ATTENTION_FUNCTIONS = NEW_ATTENTION_FUNCTIONS
    return True


def wrap_attention_forward(forward_fn: callable):
    """
    This function has been adapted from https://github.com/rachtibat/LRP-eXplains-Transformers
    Here we uniformly divide the gradients/relevances of the query, key, and value tensors by 4, 4,
    and 2 respectively. The reason is that Q, K, and V are computed via matrix matrix multiplications.
    Since there is one between Attention and V, we divide the relevance/gradient of V by 2, and since
    the Attention is constructed by the matrix multiplication of Q and K, we divide the relevance/gradient
    of Q and K by 4 (in other words 1/2 by 1/2).

    Args:
        forward_fn (callable): The original forward function of the attention module that is to be
        wrapped with the new forward function that applies the uniform rule in matmul operations via the
        Gradient*Input framework.
    """

    def attention_forward(module, query, key, value, *args, **kwargs):

        query = uniform_gradient_division_rule(query, 4)
        key = uniform_gradient_division_rule(key, 4)
        value = uniform_gradient_division_rule(value, 2)

        if "dropout" in kwargs:
            kwargs["dropout"] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)

    return attention_forward


def gated_mlp(self, x: torch.Tensor):
    """
    This function has been adapted from https://github.com/rachtibat/LRP-eXplains-Transformers
    On the element-wise non-linear activation, we apply the identity rule and
    on the element-wise multiplication, we apply the uniform rule.
    Both rules are implemented via the Gradient*Input framework.
    This function takes care of the forward pass of the gated MLP in transformer-based models,
    and it is used to patch the forward method of the gated MLP layers in these models.

    Later the activation functions is handled by the identity rule identified by the callings
    in the composites before patching takes place.

    Args:
        self (torch.nn.Module): The gated MLP module whose forward method is to be patched.
        x (torch.Tensor): The input tensor to the gated MLP module.
    Returns:
        torch.Tensor: The output tensor resulting from applying the gated MLP forward pass with the
    """
    gate_out = self.gate_proj(x)
    gate_out = self.act_fn(gate_out)

    weighted = gate_out * self.up_proj(x)
    weighted = uniform_gradient_division_rule(weighted, 2)
    return self.down_proj(weighted)


def rms_norm_forward(self, hidden_states):
    """
    This function has been adapted from https://github.com/rachtibat/LRP-eXplains-Transformers

    Check these papers for further information:
        Arras, Leila, et al. "A close look at decomposition-based XAI-methods
        for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

        Achtibat, Reduan, et al. "AttnLRP: Attention-Aware Layer-Wise Relevance
        Propagation for Transformers." Proceedings of the 41st International
        Conference on Machine Learning (2024).

    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = (
        hidden_states * torch.rsqrt(variance + self.variance_epsilon).detach()
    )

    return self.weight * hidden_states.to(input_dtype)


def layer_norm_forward(self, x):
    """
    This function has been adapted from https://github.com/rachtibat/LRP-eXplains-Transformers

    Check these papers for further information:
        Arras, Leila, et al. "A close look at decomposition-based XAI-methods
        for transformer language models." arXiv preprint arXiv:2502.15886 (2025).

        Achtibat, Reduan, et al. "AttnLRP: Attention-Aware Layer-Wise Relevance
        Propagation for Transformers." Proceedings of the 41st International
        Conference on Machine Learning (2024).
    """

    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / std.detach()
    if self.weight is not None:
        y *= self.weight
    if self.bias is not None:
        y += self.bias

    return y


##########################################
############## For ResNet ################
##########################################


def canonize_conv2d_batchnorm(
    conv2d_module: torch.nn.Conv2d, batchnorm_module: torch.nn.BatchNorm2d
):
    """
    Thie function canonizes the Conv2d and its following BatchNorm2d layers by merging
    the parameters of the BatchNorm2d layer into the Conv2d layer.

    For further information, check this paper: Pahde, Frederik, et al. "Optimizing explanations
    by network canonization and hyperparameter search." Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition. 2023.

    Args:
        conv2d_module (torch.nn.Conv2d): The Conv2d module to be canonized.
        batchnorm_module (torch.nn.BatchNorm2d): The BatchNorm2d module to be canonized.
    Returns:
        None: The function modifies the conv2d_module in-place by merging the parameters of the batchnorm_module into it.
    """
    w_bn = batchnorm_module.weight
    b_bn = batchnorm_module.bias
    s = torch.sqrt(batchnorm_module.running_var + batchnorm_module.eps)
    m = batchnorm_module.running_mean

    conv2d_module.weight = torch.nn.Parameter(
        conv2d_module.weight * (w_bn / s).view(-1, 1, 1, 1)
    )
    if conv2d_module.bias is not None:
        conv2d_module.bias = torch.nn.Parameter(
            (conv2d_module.bias - m) * (w_bn / s) + b_bn
        )
    else:
        conv2d_module.bias = torch.nn.Parameter((-m) * (w_bn / s) + b_bn)


def neutralize_batchnorm(batchnorm_module: torch.nn.BatchNorm2d):
    """
    Neutralize the BatchNorm2d module by setting its weight to 1, bias to 0,
    running mean to 0, and running variance to 1.

    Args:
        batchnorm_module (torch.nn.BatchNorm2d): The BatchNorm2d module to
    """
    batchnorm_module.weight = torch.nn.Parameter(
        torch.ones_like(batchnorm_module.weight)
    )
    batchnorm_module.bias = torch.nn.Parameter(torch.zeros_like(batchnorm_module.bias))
    batchnorm_module.running_mean = torch.zeros_like(batchnorm_module.running_mean)
    batchnorm_module.running_var = torch.ones_like(batchnorm_module.running_var)


def merge_conv2d_batchnorm(model: torch.nn.Module):
    """
    Get all Conv2d layers in the model and check if they are followed by a BatchNorm2d layer.
    If they are, canonize the Conv2d and BatchNorm2d layers by merging the parameters of the
    BatchNorm2d layer into the Conv2d layer and then neutralize the BatchNorm2d layer.

    Args:
        model (torch.nn.Module): The model whose Conv2d and BatchNorm2d layers
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            next_name = name.rsplit(".", 1)[0]
            next_module = dict(model.named_modules()).get(next_name)
            if isinstance(next_module, torch.nn.BatchNorm2d):
                canonize_conv2d_batchnorm(module, next_module)
                neutralize_batchnorm(next_module)
                print(f"Canonize Conv2d and BatchNorm2d layers: {name} and {next_name}")


def canonize_resnet(model: torch.nn.Module):
    """
    Canonize the ResNet model by merging its Conv2d and following BatchNorm2d layers.

    Args:
        model (torch.nn.Module): The ResNet model to be canonized.
    """
    merge_conv2d_batchnorm(model)
