import torch
import copy

import lexplainet.modules as modules
from lexplainet.wrapper import get_rule_wrapper_by_module

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


"""
Fused components for ResNet
"""


def canonizer_batchnorm_after_conv2d(conv, bn):

    assert isinstance(bn, torch.nn.BatchNorm2d)
    assert isinstance(conv, torch.nn.Conv2d)

    s = torch.sqrt(bn.running_var + bn.eps)
    w = bn.weight.data
    b = bn.bias.data
    m = bn.running_mean.data
    conv.weight = torch.nn.Parameter(conv.weight.data * (w / s).reshape(-1, 1, 1, 1))

    if conv.bias is None:
        conv.bias = torch.nn.Parameter(((-m) * (w / s) + b).to(conv.weight.dtype))
    else:
        conv.bias = torch.nn.Parameter((conv.bias - m) * (w / s) + b)

    return conv


def reset_batchnorm(bn):

    assert isinstance(bn, torch.nn.BatchNorm2d)

    bnc = copy.deepcopy(bn)
    bnc.reset_parameters()

    return bnc


class BasicBlockFused(BasicBlock):
    """
    Fused BasicBlock for ResNet with Canonization support
    """

    # Taken from torchvision.models.resnet.BasicBlock
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlockFused, self).__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )

        # Used for Canonization
        self.elementwise_sum = modules.ElementwiseSum()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # Canonization
        out = self.elementwise_sum(torch.stack([out, identity], dim=0))
        out = self.relu(out)

        return out


class BottleneckFused(Bottleneck):
    """
    Fused Bottleneck for ResNet with Canonization support
    """

    # Taken from torchvision.models.resnet.Bottleneck
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BottleneckFused, self).__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )

        # Used for Canonization
        self.elementwise_sum = modules.ElementwiseSum()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # Canonization
        out = self.elementwise_sum(torch.stack([out, identity], dim=0))
        out = self.relu(out)

        return out


class ResNet_canonized(ResNet):

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet_canonized, self).__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

    def setbyname(self, name, value):
        """
        Sets a module attribute by its dot-separated name.
        Returns True if successful, False otherwise.
        """

        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split(".")
        success = iteratset(self, components, value)
        return success

    def copyfromresnet(self, net, composite):
        assert isinstance(net, ResNet)
        """
        Steps:
        Copy layers from a standard ResNet to this canonized ResNet,
        applying fusions where possible.

        Strategy:
        For each module in the source net:
            - If it's a Linear layer, copy it directly.
            - If it's a Conv2d layer, store it for potential fusion.
            - If it's a BatchNorm2d layer, fuse it with the last Conv2d layer if available,
              then wrap both layers with their respective LRP rules.
            - For other layers like ReLU, AdaptiveAvgPool2d, MaxPool2d
                present only in the target net, wrap them with their respective LRP rules.
        """
        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            print("at src_module_name", src_module_name)

            if isinstance(src_module, torch.nn.Linear):
                wrapped = get_rule_wrapper_by_module(
                    copy.deepcopy(src_module), composite
                )
                if self.setbyname(src_module_name, wrapped) == False:
                    raise Modulenotfounderror(
                        "Could not find module {src_module_name} in target net to copy"
                    )
                updated_layers_names.append(src_module_name)

            if isinstance(src_module, torch.nn.Conv2d):
                last_src_module_name = src_module_name
                last_src_module = src_module

            if isinstance(src_module, torch.nn.BatchNorm2d):
                new_module = copy.deepcopy(last_src_module)
                new_module = canonizer_batchnorm_after_conv2d(new_module, bn=src_module)

                wrapped = get_rule_wrapper_by_module(new_module, composite)

                if False == self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        f"Could not find module {last_src_module_name} in target net to copy"
                    )

                updated_layers_names.append(last_src_module_name)

                wrapped = get_rule_wrapper_by_module(
                    reset_batchnorm(src_module), composite
                )

                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(
                        f"Could not find module {src_module_name} in target net to copy"
                    )
                updated_layers_names.append(src_module_name)

        # ElementwiseSum is present only in the targetclass, so must iterate here
        for target_module_name, target_module in self.named_modules():

            if isinstance(
                target_module,
                (torch.nn.ReLU, torch.nn.AdaptiveAvgPool2d, torch.nn.MaxPool2d),
            ):
                wrapped = get_rule_wrapper_by_module(target_module, composite)

                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        f"Could not find module {target_module_name} in target net to copy"
                    )
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, modules.ElementwiseSum):

                wrapped = get_rule_wrapper_by_module(target_module, composite)
                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        f"Could not find module {target_module_name} in target net to copy"
                    )
                updated_layers_names.append(target_module_name)

        for target_module_name, target_module in self.named_modules():
            if target_module_name not in updated_layers_names:
                print("Not updated:", target_module_name)


def load_canonized_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_canonized(block, layers, **kwargs)
    if pretrained:
        raise Cannotloadmodelweightserror(
            "explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model"
        )
    return model


def ResNet18_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return load_canonized_resnet(
        "resnet18", BasicBlockFused, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def ResNet50_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return load_canonized_resnet(
        "resnet50", BottleneckFused, [3, 4, 6, 3], pretrained, progress, **kwargs
    )
