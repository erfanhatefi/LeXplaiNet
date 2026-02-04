import torch
from .inference import forward_pass


def make_gradient_output(output, predicted_class):
    gradient_output = torch.zeros_like(output).to(output.device)
    gradient_output[0, predicted_class] = output[0, predicted_class]
    return gradient_output


def explain_sample(model, input_tensor):

    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    input_tensor.requires_grad = True
    output = forward_pass(model, input_tensor, gradient_required=True)

    predicted_class = output.argmax(dim=1)

    gradient_output = make_gradient_output(output, predicted_class)

    # now compute gradients with torch.autograd
    with torch.no_grad():
        relevance = torch.autograd.grad(
            outputs=output, inputs=input_tensor, grad_outputs=gradient_output
        )[0]
        normalized_relevance = (relevance / torch.max(torch.abs(relevance))).sum(dim=1)

    return normalized_relevance, relevance
