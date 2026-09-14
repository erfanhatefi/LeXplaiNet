import torch
import types
import functools


def patch_module_forward(module: torch.nn.Module, new_forward: callable, **kwargs):
    """
    This function patches the forward method of a given module with a new forward function.
    It first checks if the module is already patched to avoid multiple patching.
    If not, it saves the original forward method as old_forward and then replaces
    the forward method with the new_forward function, binding any additional keyword arguments if provided.
    We normally set the new_forward with the rule functions defined in rules.py, and the additional
    kwargs are used to specify the parameters of the rule functions like alpha, beta, and gamma.

    Args:
        module (torch.nn.Module): The module whose forward method is to be patched.
        new_forward (callable): The new forward function to replace the original forward method.
        **kwargs: Additional keyword arguments to be passed to the new forward function. LRP rule parameters.
    """
    module.old_forward = module.forward
    # Bind extra kwargs to the new_forward using functools.partial if needed
    partial_func = functools.partial(new_forward, **kwargs)
    module.forward = types.MethodType(partial_func, module)


def check_already_patched(module: torch.nn.Module, new_module: torch.nn.Module = None):
    """
    This function checks if a module's forward method has already been patched.
    It does this by checking if the module has an attribute old_forward, which
    would indicate that the original forward method has been saved and the module has been patched.
    If new_module is provided, it also checks if the current forward method of the module is
    the same as the new_module's forward method to determine if it has already been patched with
    the same new forward function.

    Args:
        module (torch.nn.Module): The module to check for patching.
        new_module (torch.nn.Module, optional): The new module to compare against
          for checking if the module has already been patched with the same new
          forward function. Default is None.
    Returns:
        bool: True if the module is already patched, False otherwise.
    """
    if new_module is None:
        if hasattr(module, "old_forward"):
            return True
        else:
            return False
    else:
        if module.__name__ == new_module.__name__:
            return True
        return False


def undo_patch(module: torch.nn):
    """
    This function undoes the patching of a module by restoring its original forward method.
    It checks if the module has an attribute old_forward, which indicates that it has been patched.
    If it has, it restores the original forward method from old_forward and then deletes the old_forward attribute.

    Args:
        module (torch.nn.Module): The module whose patching is to be undone.
    """
    if check_already_patched(module):
        module.forward = module.old_forward
        del module.old_forward
        print(f"Undo patch for Layer: {module.__class__.__name__}")


def undo_patch_all(model):
    """
    This function undoes the patching of all modules in a model by restoring their original forward methods.
    It iterates through all modules in the model and calls undo_patch on each one.

    Args:
        model (torch.nn.Module): The model whose modules' patching is to be undone.
    """
    for module in model.modules():
        undo_patch(module)


def patch_composite(model: torch.nn.Module, composite_name: dict, log=False):
    """
    This fucntions patches the forward methods of all modules in the model given a composite
    defined by a dictionary mapping module types to their corresponding new forward functions
    and additional keyword arguments.

    Args:
        model (torch.nn.Module): The model whose modules are to be patched.
        composite_name (dict): A dictionary mapping module types to their corresponding new
        forward functions and additional keyword arguments. The keys of the dictionary are the
        types of the modules to be patched, and the values are tuples containing the new
        forward function and a dictionary of additional
        log (bool, optional): If True, prints the name of each layer being patched along with the
        rule and its parameters. Default is False.
    """
    for name, module in model.named_modules():
        if type(module) in composite_name.keys():
            rule, kwargs = composite_name[type(module)]
            if log:
                print(
                    f"Patching Layer: {type(module)} with Rule: {rule.__name__} and kwargs: {kwargs}"
                )
            patch_module_forward(module, rule, **kwargs)


def patch_specific_modules(specific_composite, log=True):
    """
    Patch specific module objects directly.

    specific_composite example:
        {
            model.classifier: (epsilon_rule, {"ignore_bias": True}),
            model.dinov2.encoder.layer[11].mlp.fc2: (
                gamma_rule,
                {"gamma": 0.25, "ignore_bias": True},
            ),
        }
    """
    patched_modules = []

    for module, rule_config in specific_composite.items():
        rule, kwargs = rule_config

        # If the global composite already patched this module,
        # restore its original forward first, then apply the specific rule.
        undo_patch(module)
        patch_module_forward(module, rule, **kwargs)

        patched_modules.append(module)

        if log:
            print(f"Specific patch: {module.__class__.__name__} -> {rule.__name__}, {kwargs}")

    return patched_modules