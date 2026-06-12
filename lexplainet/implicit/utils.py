import torch


def get_layer_name(model, module):
    """
    Get the name of a layer given its module.

    Args:
        model: The PyTorch model containing the layer.
        module: The layer module for which to find the name.
    Returns:
        The name of the layer if found, otherwise None.
    """
    for name, mod in model.named_modules():
        if mod is module:
            return name
    return None


def get_latent_relevances(layer):
    """
    Get the latent relevances for a given layer via implicit LRP rules.

    Args:
        layer: The layer for which to get relevances.
    Returns:
        The latent relevances for the layer.
    """

    def forward_hook(module, input, output):

        module.output = output
        module.output.retain_grad() if module.output.requires_grad else None

    # Register the backward hook
    forward_hook_handle = layer.register_forward_hook(forward_hook)

    return forward_hook_handle


def modify_latent_relevances_at_tensor(concept_ids, layer):
    """
    Modify the latent relevances for a given layer by masking out all but the specified concept ids.
    We mask the gradient tensor that gets propagated backward through the layer, so that only the
    specified concept ids receive relevance.

    Args:
        concept_ids: The concept ids for which to keep the relevance.
        layer: The layer for which to modify the relevances.
    Returns:
        The handle for the registered hook, which can be used to remove the hook later.
    """
    print(
        f"Modifying latent relevances for concept ids: {concept_ids} in layer: {layer}"
    )

    def mask_fct(grad):
        # grad shape: (batch_size, num_channels, ...)
        mask = torch.zeros_like(grad)
        for batch_idx in range(grad.shape[0]):
            mask[batch_idx, concept_ids] = 1
        grad = grad * mask
        return grad

    def post_forward_hook(module, input, output):
        if output.requires_grad:
            print(f"Masking grad for output with shape: {output.shape}")
            output.register_hook(mask_fct)
        return output

    handle = layer.register_forward_hook(post_forward_hook)
    return handle


def modify_latent_relevances_at_module(condition_index, layer):
    """
    Modify the latent relevances for a given layer by masking out all but the specified condition index.
    We apply the mask at the module level, meaning that we modify the gradient output of the layer during
    the backward pass, so that only the specified condition index receives relevance.

    Args:
        condition_index: The index of the condition for which to keep the relevance.
        layer: The layer for which to modify the relevances.
    Returns:
        The handle for the registered hook, which can be used to remove the hook later.
    """

    def backward_hook(module, grad_output):
        num_neurons = grad_output[0].shape[
            1
        ]  # assuming grad_output[0] has shape (batch_size, num_neurons, ...)

        if condition_index < num_neurons:
            mask = torch.zeros_like(grad_output[0])
            mask[:, condition_index, ...] = (
                1  # Keep only the relevance for the specified neuron
            )

            # want to modify the gradient_output to only keep the relevance for the specified neuron
            modified_gradient_output = tuple(mask * g for g in grad_output)
            return modified_gradient_output

    # register pre hook
    hook_handle = layer.register_full_backward_pre_hook(backward_hook)

    return hook_handle
