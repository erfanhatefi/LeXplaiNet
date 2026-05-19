import torch

from .modules import ElementwiseSum


class LRPRuleWrapper(torch.nn.Module):
    """
    Wraps a nn.Module so that its forward uses a custom torch.autograd.Function rule.

    The rule is called as:
        rule.apply(x, module, *extra_args)

    This matches Functions:
        forward(ctx, x, module, param1, param2, ...)
    """

    def __init__(
        self, module: torch.nn.Module, rule: type[torch.autograd.Function], *extra_args
    ):
        super().__init__()
        self.module = module
        self.rule = rule
        self.extra_args = extra_args  # stored as tuple, positional

    def forward(self, x, *args, **kwargs):
        # If your wrapped modules ONLY take x, keep it simple:
        # return self.rule.apply(x, self.module, *self.extra_args)

        # More general: support extra forward args/kwargs if your base module needs them.
        # But: torch.autograd.Function.apply does NOT accept kwargs.
        # So if you truly need kwargs, you must encode them differently (usually avoid).
        if kwargs:
            raise TypeError(
                "LRPWrapper: kwargs are not supported because autograd.Function.apply is positional-only."
            )

        if args:
            # Optional: allow passing additional positional args to the underlying module.
            # This only works if your rule forward signature is:
            # forward(ctx, x, module, *module_args, *extra_args)
            # Most rules DON'T do this, so keep args empty unless you design for it.
            return self.rule.apply(x, self.module, *args, *self.extra_args)

        return self.rule.apply(x, self.module, *self.extra_args)


def assert_rule_in_composite(key, composite):
    if key not in composite:
        print(
            "Found no defined rule for this module name:",
            key,
        )
        raise lrplookupnotfounderror(
            "Found no defined rule for this module name:",
            key,
        )


def get_rule_wrapper_by_module(module, composite):
    # def get_lrpwrapperformodule_resnet(module, composite):

    if isinstance(module, torch.nn.modules.activation.ReLU):

        key = "ReLU"
        assert_rule_in_composite(key, composite)
        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        return LRPRuleWrapper(module, rule=rule)

    elif isinstance(module, torch.nn.modules.activation.SiLU):

        key = "SiLU"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        return LRPRuleWrapper(module, rule=rule)

    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):

        key = "BatchNorm2d"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        return LRPRuleWrapper(module, rule=rule)

    elif isinstance(module, torch.nn.modules.linear.Linear):

        key = "Linear"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        if rule.__name__ == "linear_eps_wrapper_fct":
            return LRPRuleWrapper(module, rule, *(composite[key]["eps"],))

    elif isinstance(module, torch.nn.modules.conv.Conv2d):

        key = "Conv2d"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        if rule.__name__ == "conv2d_beta0_wrapper_fct":
            return LRPRuleWrapper(module, rule, *(composite[key]["ignore_bias"],))
        elif rule.__name__ == "conv2d_eps_wrapper_fct":
            return LRPRuleWrapper(
                module,
                rule,
                *(
                    composite[key]["ignore_bias"],
                    composite[key]["eps"],
                ),
            )
        elif rule.__name__ == "conv2d_betaany_wrapper_fct":
            return LRPRuleWrapper(
                module,
                rule,
                *(
                    composite[key]["ignore_bias"],
                    composite[key]["beta"],
                ),
            )
        elif rule.__name__ == "conv2d_betaadaptive_wrapper_fct":
            return LRPRuleWrapper(
                module,
                rule,
                *(
                    composite[key]["ignore_bias"],
                    torch.tensor(composite[key]["max_beta"]),
                ),
            )
        elif rule.__name__ == "conv2d_gammaany_wrapper_fct":
            return LRPRuleWrapper(
                module,
                rule,
                *(
                    composite[key]["ignore_bias"],
                    torch.tensor(composite[key]["gamma"]),
                ),
            )
        elif rule.__name__ == "conv2d_gammalifted_wrapper_fct":
            return LRPRuleWrapper(
                module,
                rule,
                *(
                    composite[key]["ignore_bias"],
                    torch.tensor(composite[key]["gamma"]),
                ),
            )
        else:
            print(rule, rule.__name__)
            print(
                "Unknown Rule for Conv2d",
                rule,
                key,
                rule.__name__,
            )
            exit()

    elif isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
        key = "AdaptiveAvgPool2d"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        return LRPRuleWrapper(module, rule, *(composite[key]["eps"],))

    elif isinstance(module, torch.nn.MaxPool2d):

        key = "MaxPool2d"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        return LRPRuleWrapper(module, rule)

    elif isinstance(module, ElementwiseSum):  # resnet specific

        key = "ElementwiseSum"
        assert_rule_in_composite(key, composite)

        print(f"Wrap {key}")

        rule = composite[key]["rule"]
        return LRPRuleWrapper(module, rule, *(composite[key]["eps"],))

    else:
        print(
            "Food no lookup for this module:",
            module,
            type(module),
            type(module).__name__,
        )
        raise lrplookupnotfounderror("Found no lookup for this module:", module)
