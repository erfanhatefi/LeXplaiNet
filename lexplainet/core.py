import torch


class LRPBackwardPass:
    """
    Class to perform LRP backward pass through a given layer/module
    """

    @staticmethod
    def safe_divide(numerator, divisor, eps0, eps):
        """
        Docstring for safe_divide

        This function performs a safe division operation to avoid division by zero.
        This mechanism is more stable than adding a small constant to the denominator.
        Here, eps0 is added when the divisor is exactly zero, and eps scaled by the
        sign of the divisor is added otherwise.

        It keeps the denominator non-zerowithout biasing the magnitude much when the divisor is small.
        The operation can be summarized as:
            \text{denom}' = d \;+\; \varepsilon_0 \cdot \mathbf{1}[d=0] \;+\; \varepsilon \cdot \operatorname{sign}(d)

        Additionally, it preserves the sign:

            With d + eps, if d is negative and close to zero, adding a positive eps moves it toward 0 and can even cross it:
            d = -1e-12, eps = 1e-6 ⇒ d + eps ≈ +1e-6 (sign flips!)

            With eps * sign(d), you add -eps for negative d, +eps for positive d, so the sign is preserved:
            d = -1e-12 ⇒ d - eps ≈ -1e-6 (no flip)

        The statbilization is not one-sided (as a bias) around zero, but both the positive and negative
          denominators asymmetrically get near zero on both sides

        The flip in sign can lead to incorrect attributions in LRP, and also computation of gradients,
        therefore this stablization is preferred.

        :param numerator: Description
        :param divisor: Description
        :param eps0: Description
        :param eps: Description
        """
        return numerator / (
            divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign()
        )

    @staticmethod
    def lrp_backward(module_input, layer, relevance_output, eps0, eps):
        """
        The implementation of LRP backward pass through a given layer/module.

        The formulation used is:
            R^{in} = x \;\odot\; (J^\top S)

            where S = R^{out} \;/\; z + (stabilizer)
            and J corresponds to the Jacobian of the layer's forward pass
            noted by J = \frac{\partial z}{\partial x}. Additionally z
            is the output of the layer/module during the forward pass.

            In PyTroch, J^\top S is considered as a Vector-Jacobian Product (VJP),
            and can be implementad by performing a backward pass on z with S as the gradient.


        """

        # zeroing out the gradients of the module's/layer's input_
        if module_input.grad is not None:
            module_input.grad.zero_()

        # detach the relevance output, as it is not part of the computational graph
        # and is already propagated backwards
        relevance_output_data = relevance_output.clone().detach()

        # compute the output of the layer/module during forward pass
        with torch.enable_grad():
            Z = layer(module_input)

        # compute S = R^{out} / (z + stabilizer)
        # represents a normalized relevance based on the layer's output
        S = LRPBackwardPass.safe_divide(
            relevance_output_data, Z.clone().detach(), eps0, eps
        )

        # compute Vector-Jacobian Product J^T S via backward pass
        Z.backward(S)

        # compute the input_ relevance R^{in} = x * (J^T S)
        # as a element-wise product of the input_ and the computed VJP (modified gradient in this case)
        relevance_input = module_input.data * module_input.grad.data
        return relevance_input
