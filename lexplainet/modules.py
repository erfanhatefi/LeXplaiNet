import torch


class posnegconv(torch.nn.Module):
    """
    Implementation of the PosNegConv layer
    This module clones a given Conv2d module into two separate Conv2d modules:
    one for positive weights and another for negative weights.
    During the forward pass, it applies the positive convolution to the
    positive part of the input and the negative convolution to the negative
    part of the input, and then sums the results.
    Elaborately, the output of the two generated convolutions are always
    positive as they are the cases of positive weights * positive inputs,
    either negative weights * negative inputs.

    ATTENTION: Please note that because of the output being always positive, the forward pass
    of this module if not the same as the original Conv2d module!
    """

    def _clone_module(self, module):
        """
        To clone a Conv2d module structure without copying weights/biases

        :param module: a torch.nn.Conv2d module
        :return: a cloned torch.nn.Conv2d module with same structure but uninitialized weights
        """
        clone = torch.nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{
                attr: getattr(module, attr)
                for attr in ["stride", "padding", "dilation", "groups"]
            }
        )
        return clone.to(module.weight.device)

    def __init__(self, conv, ignore_bias_flag):
        super(posnegconv, self).__init__()

        self.posconv = self._clone_module(conv)
        # Get only positive weights
        self.posconv.weight = torch.nn.Parameter(
            conv.weight.data.clone().clamp(min=0)
        ).to(conv.weight.device)

        self.negconv = self._clone_module(conv)
        # Get only negative weights
        self.negconv.weight = torch.nn.Parameter(
            conv.weight.data.clone().clamp(max=0)
        ).to(conv.weight.device)

        if ignore_bias_flag == True:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                # Get positive and negative biases
                self.posconv.bias = torch.nn.Parameter(
                    conv.bias.data.clone().clamp(min=0)
                )
                self.negconv.bias = torch.nn.Parameter(
                    conv.bias.data.clone().clamp(max=0)
                )

    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn


class invertedposnegconv(torch.nn.Module):
    """
    This module is similar to posnegconv, but inverts the treatment of
    positive and negative inputs. Specifically, it applies the positive
    weight to the negative part of the input and the negative
    weight to the positive part of the input.

    ATTENTION: Please note that because of the output being always negative or zero, the forward pass
    of this module if not the same as the original Conv2d module!
    """

    def _clone_module(self, module):
        clone = torch.nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{
                attr: getattr(module, attr)
                for attr in ["stride", "padding", "dilation", "groups"]
            }
        )
        return clone.to(module.weight.device)

    def __init__(self, conv, ignore_bias_flag):
        super(invertedposnegconv, self).__init__()

        self.posconv = self._clone_module(conv)
        # Get only positive weights
        self.posconv.weight = torch.nn.Parameter(
            conv.weight.data.clone().clamp(min=0)
        ).to(conv.weight.device)

        self.negconv = self._clone_module(conv)
        # Get only negative weights
        self.negconv.weight = torch.nn.Parameter(
            conv.weight.data.clone().clamp(max=0)
        ).to(conv.weight.device)

        self.posconv.bias = None
        self.negconv.bias = None
        if ignore_bias_flag == False:
            if conv.bias is not None:
                # Get positive and negative biases
                self.posconv.bias = torch.nn.Parameter(
                    conv.bias.data.clone().clamp(min=0)
                )
                self.negconv.bias = torch.nn.Parameter(
                    conv.bias.data.clone().clamp(max=0)
                )

    def forward(self, x):
        # Positive weights on negatives inputs
        vp = self.posconv(torch.clamp(x, max=0))  # on negatives
        # Negative weights on positive inputs
        vn = self.negconv(torch.clamp(x, min=0))  # on positives

        return vp + vn  # zero or neg


class gammaconv(torch.nn.Module):
    """
    Note on the implementation of gamma rule:
    The gamma rule can be seen via this formular:
    R_j =
    \sum_k
    \frac{a_j \, (w_{jk} + \gamma\, w_{jk}^{+})}
    {\sum_{j'} a_{j'} \, (w_{j'k} + \gamma\, w_{j'k}^{+})}
    \; R_k
    A non-precise formulization is following this approach:
    (z_j + (\gamma \times z_j^{+}) / (z_k + \gamma z_k^{+} )) * R_k
    In other words, the output of the positive contributions is scaled by (1 + gamma).
    (output + gamma * positive_output)

    Since the positive part is considered as a more stable contribution and we don't want
    to touch/modify it, there is a methematical trick where we multiplu the original formula
    by \frac{1}{1 + \gamma} both in numerator and denominator, leading to:
    1/gamma * output + positive_output * gamma * (1/gamma) = output / gamma + positive_output

    Either of the implemnetations are equivalent, but the latter is more stable numerically
    """

    def _clone_module(self, module):
        clone = torch.nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{
                attr: getattr(module, attr)
                for attr in ["stride", "padding", "dilation", "groups"]
            }
        )
        return clone.to(module.weight.device)

    def __init__(self, conv, ignore_bias_flag, gamma):
        super().__init__()

        self.gamma = gamma
        # Create a module for positive convolution
        self.pnconv = posnegconv(conv, ignore_bias_flag)
        self.conv = self._clone_module(conv)

        if ignore_bias_flag == True:
            self.conv.bias = None
        else:
            if conv.bias is not None:
                self.conv.bias = torch.nn.Parameter(conv.bias.data.clone())

    def forward(self, x):
        # Positive convulation output
        vp = self.pnconv(x)
        # Original convolution output
        u = self.conv(x)

        return vp + u / self.gamma


class gammaconv_lifted(torch.nn.Module):
    """
    This is a more stablized version of the gamma rule where you
    lift the gamma values so that they satisfy an important stabilization condition
    for the gamma rule where states:

    gamma ^ (-1/2) * N < P

    where N is the negative part magnitude and P is the positive part magnitude.
    (P := \sum_{b:\, w_{ab}z_b > 0} w_{ab}z_b, and N := \sum_{b:\, w_{ab}z_b < 0} (-w_{ab}z_b)
    \sum_b w_{ab}z_b = \sum_{b: w_{ab}z_b>0} w_{ab}z_b + \sum_{b: w_{ab}z_b<0} w_{ab}z_b =
    P + (-N) = P - N)
    You can modify this condition as so:

    gamma ^ (-1/2) < P / N
    gamma ^ -1 < (P / N) ^ 2
    gamma >= (N / P) ^ 2

    Therefore, in this implementation, we ensure that gamma is at least (N / P) ^ 2.
    This is also done by:

    u = P - N
    P = vp

    then 1 - u / vp = 1 - (P - N) / P = N / P
    therefore, (1 - u / vp) ^ 2 = (N / P) ^ 2
    and we ensure that gamma is at least this value.
    """

    def _clone_module(self, module):
        clone = torch.nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{
                attr: getattr(module, attr)
                for attr in ["stride", "padding", "dilation", "groups"]
            }
        )
        return clone.to(module.weight.device)

    def __init__(self, conv, ignore_bias_flag, gamma):
        super().__init__()

        self.gamma = gamma
        # Create a module for positive convolution
        self.pnconv = posnegconv(conv, ignore_bias_flag)
        self.conv = self._clone_module(conv)

        if ignore_bias_flag == True:
            self.conv.bias = None
        else:
            if conv.bias is not None:
                self.conv.bias = torch.nn.Parameter(conv.bias.data.clone())

    def forward(self, x):
        # Positive convulation output
        vp = self.pnconv(x)
        # Original convolution output
        u = self.conv(x)

        epsthresh = 1e-10
        vpstab = torch.where(vp > epsthresh, vp, epsthresh)
        mingamma = (1.0 - u / vpstab) ** 2

        return vp + u / torch.maximum(self.gamma, mingamma)
