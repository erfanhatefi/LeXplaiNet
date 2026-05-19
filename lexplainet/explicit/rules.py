import torch
from .core import LRPBackwardPass
from .modules import (
    ElementwiseSum,
    posnegconv,
    invertedposnegconv,
    gammaconv,
    gammaconv_lifted,
)
from .utils import configvalues_totensorlist, tensorlist_todict


class relu_wrapper_fct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module):

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None


# not efficient implementation which stashes the
# whole module via ctx.module = module
# class linearlayer_eps_wrapper_fct(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, module, eps):
#         ctx.save_for_backward(x)
#         ctx.module = module
#         ctx.eps

#         return module.forward(x)

#     @staticmethod
#     def backward(ctx, grad_output):

#         module = ctx.module
#         eps = ctx.eps
#         X = ctx.saved_tensors[0].clone().detach().requires_grad_(True)

#         R = LRPBackwardPass.lrp_backward(
#             _input=X, layer=module, relevance_output=grad_output, eps0=eps, eps=eps
#         )

#         return R, None, None


# lineareps_wrapper_fct
class linear_eps_wrapper_fct(torch.autograd.Function):

    layer_property_names = ["in_features", "out_features"]

    @staticmethod
    def forward(ctx, x, module, eps):
        values = configvalues_totensorlist(
            module,
            propertynames=linear_eps_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )
        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, epstensor, *values
        )  # *values unpacks the list

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, epstensor, *values = ctx.saved_tensors

        paramsdict = tensorlist_todict(
            values, propertynames=linear_eps_wrapper_fct.layer_property_names
        )

        if bias is None:
            module = torch.nn.Linear(**paramsdict, bias=False)
        else:
            module = torch.nn.Linear(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(bias)

        module.weight = torch.nn.Parameter(weight)

        eps = epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=module,
            relevance_output=grad_output,
            eps0=eps,
            eps=eps,
        )

        return R, None, None


class eltwisesum_eps_wrapper_fct(
    torch.autograd.Function
):  # to be used with generic_activation_pool_wrapper_class(module,this)
    @staticmethod
    def forward(ctx, stackedx, module, eps):
        epstensor = torch.tensor([eps], dtype=torch.float32, device=stackedx.device)

        ctx.save_for_backward(stackedx, epstensor)

        return module.forward(stackedx)

    @staticmethod
    def backward(ctx, grad_output):
        stackedx, epstensor = ctx.saved_tensors

        # X0 = x1.clone().detach() #.requires_grad_(True)
        # X1 = x2.clone().detach() #.requires_grad_(True)
        # X=torch.stack([X0, X1], dim=0) # along a new dimension!

        X = stackedx.clone().detach().requires_grad_(True)

        eps = epstensor.item()

        module = ElementwiseSum().to(X.device)

        Rtmp = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=module,
            relevance_output=grad_output,
            eps0=eps,
            eps=eps,
        )

        return Rtmp, None, None


# Why do we even have this?
# Claims from Zennit:
# For LRP, the gradient of MaxPool assigns values only to the largest inputs (winner-takes-all),
#  which is already the intended behaviour for LRP rules. Other operations for which the gradient
#  is already the intended behaviour for LRP are, for example, constant padding, concatenation,
#  cropping, indexing and slicing.
class maxpool2d_wrapper_fct(torch.autograd.Function):

    layer_property_names = [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ]

    @staticmethod
    def forward(ctx, x, module):
        values = configvalues_totensorlist(
            module=module,
            propertynames=maxpool2d_wrapper_fct.layer_property_names,
            device=x.device,
        )

        ctx.save_for_backward(x, *values)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, *values = ctx.saved_tensors

        paramsdict = tensorlist_todict(
            values,
            propertynames=maxpool2d_wrapper_fct.layer_property_names,
        )

        module = torch.nn.MaxPool2d(**paramsdict)

        X = input_.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            Z = module.forward(X)

        relevance_output_data = grad_output.clone().detach()
        Z.backward(relevance_output_data)

        R = X.grad

        return R, None


class adaptiveavgpool2d_wrapper_fct(torch.autograd.Function):

    layer_property_names = ["output_size"]

    @staticmethod
    def forward(ctx, x, module, eps):
        values = configvalues_totensorlist(
            module,
            propertynames=adaptiveavgpool2d_wrapper_fct.layer_property_names,
            device=x.device,
        )

        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)

        ctx.save_for_backward(x, epstensor, *values)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, epstensor, *values = ctx.saved_tensors

        paramsdict = tensorlist_todict(
            values, propertynames=adaptiveavgpool2d_wrapper_fct.layer_property_names
        )

        eps = epstensor.item()

        module = torch.nn.AdaptiveAvgPool2d(**paramsdict)

        X = input_.clone().detach().requires_grad_(True)

        R = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=module,
            relevance_output=grad_output,
            eps0=eps,
            eps=eps,
        )
        return R, None, None


class conv2d_eps_wrapper_fct(torch.autograd.Function):
    """
    Implementation of LRP Beta=0 rule (or ZPlus) for Conv2d layers.
    """

    layer_property_names = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    @staticmethod
    def forward(ctx, x, module, ignore_bias_flag, eps):
        values = configvalues_totensorlist(
            module,
            propertynames=conv2d_beta0_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )

        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ignore_bias_tensor = torch.tensor(
            [ignore_bias_flag], dtype=torch.bool, device=module.weight.device
        )

        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, epstensor, ignore_bias_tensor, *values
        )

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, epstensor, ignore_bias_tensor, *values = (
            ctx.saved_tensors
        )

        paramsdict = tensorlist_todict(
            values, propertynames=conv2d_beta0_wrapper_fct.layer_property_names
        )

        if conv2dbias is None:
            module = torch.nn.Conv2d(**paramsdict, bias=False)
        else:
            module = torch.nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        eps = epstensor.item()
        X = input_.clone().detach().requires_grad_(True)

        R = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=module,
            relevance_output=grad_output,
            eps0=eps,
            eps=eps,
        )

        return R, None, None, None


class conv2d_beta0_wrapper_fct(torch.autograd.Function):
    """
    Implementation of LRP Beta=0 rule (or ZPlus) for Conv2d layers.
    """

    layer_property_names = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    @staticmethod
    def forward(ctx, x, module, ignore_bias_flag):
        values = configvalues_totensorlist(
            module,
            propertynames=conv2d_beta0_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ignore_bias_tensor = torch.tensor(
            [ignore_bias_flag], dtype=torch.bool, device=module.weight.device
        )

        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, ignore_bias_tensor, *values
        )

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, ignore_bias_tensor, *values = (
            ctx.saved_tensors
        )

        paramsdict = tensorlist_todict(
            values, propertynames=conv2d_beta0_wrapper_fct.layer_property_names
        )

        if conv2dbias is None:
            module = torch.nn.Conv2d(**paramsdict, bias=False)
        else:
            module = torch.nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        pnconv = posnegconv(module, ignore_bias_flag=ignore_bias_tensor.item())

        X = input_.clone().detach().requires_grad_(True)

        R = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=pnconv,
            relevance_output=grad_output,
            eps0=1e-12,
            eps=0,
        )

        return R, None, None


class conv2d_betaany_wrapper_fct(torch.autograd.Function):
    """
    This is the implemenation of LRP Beta rule (or AlphaBeta) for Conv2d layers.
    The relevance is computed as:
    R = (1 + beta) * R_positive - beta * R_negative

    In Zennit e.g. this is called AlphaBeta rule with alpha=1+beta:
    R = alpha * R_positive - beta * R_negative
    """

    layer_property_names = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    @staticmethod
    def forward(ctx, x, module, ignore_bias_flag, beta):
        values = configvalues_totensorlist(
            module,
            propertynames=conv2d_betaany_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ignore_bias_tensor = torch.tensor(
            [ignore_bias_flag], dtype=torch.bool, device=module.weight.device
        )

        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, ignore_bias_tensor, beta, *values
        )

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, ignore_bias_tensor, beta, *values = (
            ctx.saved_tensors
        )

        paramsdict = tensorlist_todict(
            values, propertynames=conv2d_betaany_wrapper_fct.layer_property_names
        )

        if conv2dbias is None:
            module = torch.nn.Conv2d(**paramsdict, bias=False)
        else:
            module = torch.nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        # Conv2d with positive output
        pnconv = posnegconv(module, ignore_bias_flag=ignore_bias_tensor.item())
        # Conv2d with negative output
        invertedpnconv = invertedposnegconv(
            module, ignore_bias_flag=ignore_bias_tensor.item()
        )

        X = input_.clone().detach().requires_grad_(True)

        # Relevance of positive contributions
        R_positive = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=pnconv,
            relevance_output=grad_output,
            eps0=1e-12,
            eps=0,
        )
        # Relevance of negative contributions
        R_negative = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=invertedpnconv,
            relevance_output=grad_output,
            eps0=-1e-12,
            eps=0,
        )

        R = (1 + beta) * R_positive - beta * R_negative
        return R, None, None, None


class conv2d_betaadaptive_wrapper_fct(torch.autograd.Function):
    """
    Implementation of AdaptiveBeta rule for Conv2d layers.

    Explanation on setting beta per output neuron:
    if output(x) is negative; then beta is considered as zero and per output this
    will behave like beta=0 rule (ZPlus).

    if output(x) is positive; then beta is set as: min( neg_contributions(x) / output(x) , maxbeta )
    where neg_contributions(x) is the scale of relevance propagated through the negative contributions.
    In such case, if the negative contributions are very high compared to the output, then beta
    will be high (up to maxbeta) and more relevance will be subtracted through the negative contributions.
    The heatmap in this case looks more contrastive.

    In other case where negative contributions are low compared to the output, beta will be low (close to zero).
    """

    layer_property_names = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    @staticmethod
    def forward(ctx, x, module, ignore_bias_flag, maxbeta):
        values = configvalues_totensorlist(
            module,
            propertynames=conv2d_betaadaptive_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        ignore_bias_tensor = torch.tensor(
            [ignore_bias_flag], dtype=torch.bool, device=module.weight.device
        )
        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, ignore_bias_tensor, maxbeta, *values
        )

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, ignore_bias_tensor, maxbeta, *values = (
            ctx.saved_tensors
        )

        paramsdict = tensorlist_todict(
            values, propertynames=conv2d_betaadaptive_wrapper_fct.layer_property_names
        )

        if conv2dbias is None:
            module = torch.nn.Conv2d(**paramsdict, bias=False)
        else:
            module = torch.nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        # Conv2d with positive output
        pnconv = posnegconv(module, ignore_bias_flag=ignore_bias_tensor.item())
        # Conv2d with negative output
        invertedpnconv = invertedposnegconv(
            module, ignore_bias_flag=ignore_bias_tensor.item()
        )

        # betatensor =  -neg / conv but care for zeros
        X = input_.clone().detach()
        out = module(X)
        # magnitude of negative contributions
        negscores = -invertedpnconv(X)

        # assign the ratio per output position
        betatensor = torch.zeros_like(out)
        betatensor[out > 0] = torch.minimum(
            negscores[out > 0] / out[out > 0], maxbeta.to(out.device)
        )

        X.requires_grad_(True)

        # Note that the computation of LRP cannot be made as below:
        # R_positive = LRPBackwardPass.lrp_backward(
        #     _input=X, layer=pnconv, relevance_output=grad_output, eps0=1e-12, eps=0
        # )
        # # Relevance of negative contributions
        # R_negative = LRPBackwardPass.lrp_backward(
        #     _input=X,
        #     layer=invertedpnconv,
        #     relevance_output=grad_output,
        #     eps0=-1e-12,
        #     eps=0,
        # )
        # R = (1 + beta) * R_positive - beta * R_negative
        # because beta is not a constant which gets broadcasted, but
        # a tensor at the dimension of the output of the layer.

        R1 = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=pnconv,
            relevance_output=grad_output * (1 + betatensor),
            eps0=1e-12,
            eps=0,
        )
        R2 = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=invertedpnconv,
            relevance_output=grad_output * betatensor,
            eps0=1e-12,
            eps=0,
        )

        R = R1 - R2
        return R, None, None, None


class conv2d_gammaany_wrapper_fct(torch.autograd.Function):

    layer_property_names = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    @staticmethod
    def forward(ctx, x, module, ignore_bias_flag, gamma):
        values = configvalues_totensorlist(
            module,
            propertynames=conv2d_gammaany_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ignore_bias_tensor = torch.tensor(
            [ignore_bias_flag], dtype=torch.bool, device=module.weight.device
        )
        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, ignore_bias_tensor, gamma, *values
        )

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, ignore_bias_tensor, gamma, *values = (
            ctx.saved_tensors
        )

        paramsdict = tensorlist_todict(
            values, propertynames=conv2d_gammaany_wrapper_fct.layer_property_names
        )

        if conv2dbias is None:
            module = torch.nn.Conv2d(**paramsdict, bias=False)
        else:
            module = torch.nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        gaconv = gammaconv(
            module, ignore_bias_flag=ignore_bias_tensor.item(), gamma=gamma
        )

        X = input_.clone().detach().requires_grad_(True)
        R = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=gaconv,
            relevance_output=grad_output,
            eps0=1e-12,
            eps=1e-10,
        )

        return R, None, None, None


class conv2d_gammalifted_wrapper_fct(torch.autograd.Function):

    layer_property_names = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    @staticmethod
    def forward(ctx, x, module, ignore_bias_flag, gamma):
        values = configvalues_totensorlist(
            module,
            propertynames=conv2d_gammalifted_wrapper_fct.layer_property_names,
            device=module.weight.device,
        )

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        ignore_bias_tensor = torch.tensor(
            [ignore_bias_flag], dtype=torch.bool, device=module.weight.device
        )

        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, ignore_bias_tensor, gamma, *values
        )

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, ignore_bias_tensor, gamma, *values = (
            ctx.saved_tensors
        )

        paramsdict = tensorlist_todict(
            values, propertynames=conv2d_gammalifted_wrapper_fct.layer_property_names
        )

        if conv2dbias is None:
            module = torch.nn.Conv2d(**paramsdict, bias=False)
        else:
            module = torch.nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        gaconv = gammaconv_lifted(
            module, ignore_bias_flag=ignore_bias_tensor.item(), gamma=gamma
        )

        X = input_.clone().detach().requires_grad_(True)
        R = LRPBackwardPass.lrp_backward(
            module_input=X,
            layer=gaconv,
            relevance_output=grad_output,
            eps0=1e-12,
            eps=1e-10,
        )

        return R, None, None, None
