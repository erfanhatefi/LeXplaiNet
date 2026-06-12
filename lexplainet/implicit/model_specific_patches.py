import torch
from .composite_core import check_already_patched
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
    conv2d_module: torch.nn.Module,
    batchnorm_module: torch.nn.modules.batchnorm._BatchNorm,
):
    """
    Thie function canonizes a Conv2d/Linear layer and its following BatchNorm layer by
    merging the parameters of the BatchNorm layer into the preceding linear layer.

    For further information, check this paper: Pahde, Frederik, et al. "Optimizing explanations
    by network canonization and hyperparameter search." Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition. 2023.

    Args:
        conv2d_module (torch.nn.Module): Conv2d or Linear module to be canonized.
        batchnorm_module (torch.nn.modules.batchnorm._BatchNorm): BatchNorm module to be canonized.
    Returns:
        None: The function modifies conv2d_module in-place by merging the parameters of batchnorm_module into it.
    """
    if not isinstance(conv2d_module, (torch.nn.Conv2d, torch.nn.Linear)):
        raise TypeError(
            f"Expected Conv2d or Linear for merge, got {type(conv2d_module).__name__}."
        )

    w_bn = batchnorm_module.weight
    b_bn = batchnorm_module.bias
    s = torch.sqrt(batchnorm_module.running_var + batchnorm_module.eps)
    m = batchnorm_module.running_mean

    scale = (w_bn / s).view(-1, *([1] * (conv2d_module.weight.ndim - 1)))

    conv2d_module.weight = torch.nn.Parameter(conv2d_module.weight * scale)
    if conv2d_module.bias is not None:
        conv2d_module.bias = torch.nn.Parameter(
            (conv2d_module.bias - m) * (w_bn / s) + b_bn
        )
    else:
        conv2d_module.bias = torch.nn.Parameter((-m) * (w_bn / s) + b_bn)


def neutralize_batchnorm(batchnorm_module: torch.nn.modules.batchnorm._BatchNorm):
    """
    Neutralize the BatchNorm module by setting its weight to 1, bias to 0,
    running mean to 0, and running variance to 1.

    Args:
        batchnorm_module (torch.nn.modules.batchnorm._BatchNorm): The BatchNorm module to
    """
    if batchnorm_module.weight is not None:
        batchnorm_module.weight = torch.nn.Parameter(
            torch.ones_like(batchnorm_module.weight)
        )
    if batchnorm_module.bias is not None:
        batchnorm_module.bias = torch.nn.Parameter(
            torch.zeros_like(batchnorm_module.bias)
        )
    batchnorm_module.running_mean = torch.zeros_like(batchnorm_module.running_mean)
    batchnorm_module.running_var = torch.ones_like(batchnorm_module.running_var)
    batchnorm_module.eps = 0.0


def merge_conv2d_batchnorm(model: torch.nn.Module):
    """
    Get all Conv2d layers in the model and check if they are followed by a BatchNorm2d layer.
    If they are, canonize the Conv2d and BatchNorm2d layers by merging the parameters of the
    BatchNorm2d layer into the Conv2d layer and then neutralize the BatchNorm2d layer.

    Args:
        model (torch.nn.Module): The model whose Conv2d and BatchNorm2d layers
    """

    def collect_leaves(module):
        """Depth-first, in-order leaf traversal (same adjacency logic as Zennit)."""
        is_leaf = True
        for child in module.children():
            is_leaf = False
            yield from collect_leaves(child)
        if is_leaf:
            yield module

    def is_linear_leaf(module):
        # In this project we treat Conv2d and Linear as the mergeable linear layers.
        return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))

    def is_batchnorm_leaf(module):
        return isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ),
        )

    module_to_name = {id(module): name for name, module in model.named_modules()}

    last_leaf = None
    for leaf in collect_leaves(model):
        if is_linear_leaf(last_leaf) and is_batchnorm_leaf(leaf):
            canonize_conv2d_batchnorm(last_leaf, leaf)
            neutralize_batchnorm(leaf)
            print(
                "Canonize linear and BatchNorm layers: "
                f"{module_to_name.get(id(last_leaf), '<unnamed>')} and {module_to_name.get(id(leaf), '<unnamed>')}"
            )
        last_leaf = leaf

    # Old version 1 (name-based lookup, kept for reference)
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         next_name = name.rsplit(".", 1)[0]
    #         next_module = dict(model.named_modules()).get(next_name)
    #         if isinstance(next_module, torch.nn.BatchNorm2d):
    #             canonize_conv2d_batchnorm(module, next_module)
    #             neutralize_batchnorm(next_module)
    #             print(f"Canonize Conv2d and BatchNorm2d layers: {name} and {next_name}")

    # Old version 2 (adjacent-module indexing, kept for reference)
    # modules = list(model.named_modules())
    # for idx, (name, module) in enumerate(modules[:-1]):
    #     if not isinstance(module, torch.nn.Conv2d):
    #         continue
    #     next_name, next_module = modules[idx + 1]
    #     if isinstance(next_module, torch.nn.BatchNorm2d):
    #         canonize_conv2d_batchnorm(module, next_module)
    #         neutralize_batchnorm(next_module)
    #         print(f"Canonize Conv2d and BatchNorm2d layers: {name} and {next_name}")

    # Old version 3 (last-seen Conv2d in named_modules traversal, kept for reference)
    # last_conv_name = None
    # last_conv_module = None
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         last_conv_name = name
    #         last_conv_module = module
    #         continue
    #     if isinstance(module, torch.nn.BatchNorm2d) and last_conv_module is not None:
    #         canonize_conv2d_batchnorm(last_conv_module, module)
    #         neutralize_batchnorm(module)
    #         print(f"Canonize Conv2d and BatchNorm2d layers: {last_conv_name} and {name}")
    #         last_conv_name = None
    #         last_conv_module = None


def canonize_vgg_bn(model: torch.nn.Module):
    """
    Canonize the VGG-BN model by merging its Conv2d and following BatchNorm2d layers.

    Args:
        model (torch.nn.Module): The ResNet model to be canonized.
    """
    merge_conv2d_batchnorm(model)


def canonize_resnet(model: torch.nn.Module):
    """
    Canonize the ResNet model by merging its Conv2d and following BatchNorm2d layers.

    Args:
        model (torch.nn.Module): The ResNet model to be canonized.
    """
    merge_conv2d_batchnorm(model)


def canonize_efficientnet(model: torch.nn.Module):
    """
    Canonize the EfficientNet model by merging its Conv2d and following BatchNorm2d layers.

    Args:
        model (torch.nn.Module): The EfficientNet model to be canonized.
    """
    merge_conv2d_batchnorm(model)



def se_block_gate_forward(self, input):
    """
    For EfficientNet, we need to patch the forward function of the gate in the Squeeze-and-Excitation
    block to apply the uniform rule on the element-wise multiplication via the Gradient*Input framework.
    This is because the gate in the Squeeze-and-Excitation block is an element-wise multiplication operation,
    and we want to ensure that the relevance scores are correctly propagated through this operation according
    to the uniform rule.
    
    Block forward function for the gate in the Squeeze-and-Excitation block.
    We apply the identity rule on the activation function later when we identify
    it in the composite before patching takes place, and here we apply the uniform 
    rule on the element-wise multiplication via the Gradient*Input framework.

    Args:
        self (torch.nn.Module): The Squeeze-and-Excitation block module whose forward method is to be patched.
        input (torch.Tensor): The input tensor to the Squeeze-and-Excitation block module.
    Returns:
        torch.Tensor: The output tensor resulting from applying the block forward pass with the uniform rule on
    """
    scale = self._scale(input).detach()
    return scale * input

def se_uniform_product_forward(self, input):
    """
    For EfficientNet, we need to patch the forward function of the
    gate in the Squeeze-and-Excitation block to apply the uniform rule 
    on the element-wise multiplication via the Gradient*Input framework. 
    This is because the gate in the Squeeze-and-Excitation block is an element-wise 
    multiplication operation, and we want to ensure that the relevance scores are correctly 
    propagated through this operation according to the uniform rule.

    Uniform product forward function for the Squeeze-and-Excitation block.
    We apply the uniform rule on the element-wise multiplication via the Gradient*Input framework.

    Args:
        self (torch.nn.Module): The Squeeze-and-Excitation block module whose forward method is to be patched.
        input (torch.Tensor): The input tensor to the Squeeze-and-Excitation block module.
    Returns:
        torch.Tensor: The output tensor resulting from applying the block forward pass with the uniform rule on
    """
    scale = self._scale(input)
    scale = uniform_gradient_division_rule(scale, 2)
    input = uniform_gradient_division_rule(input, 2)
    return scale * input