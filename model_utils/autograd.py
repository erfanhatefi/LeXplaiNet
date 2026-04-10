import torch
from .inference import forward_pass


def make_gradient_output(output, predicted_class):
    gradient_output = torch.zeros_like(output).to(output.device)
    gradient_output[0, predicted_class] = output[0, predicted_class]
    return gradient_output


def explain_model_prediction(model, input_tensor):

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


def explain_output(model, input_tensor, output_index):

    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    input_tensor.requires_grad = True
    output = forward_pass(model, input_tensor, gradient_required=True)

    if output_index is None:
        predicted_class = output.argmax(dim=1)
        gradient_output = make_gradient_output(output, predicted_class)
    elif output_index == "logit":
        gradient_output = output
    elif output_index == "ones":
        gradient_output = torch.ones_like(output).to(output.device)
    else:
        predicted_class = output_index
        gradient_output = make_gradient_output(output, predicted_class)

    # now compute gradients with torch.autograd
    with torch.no_grad():
        relevance = torch.autograd.grad(
            outputs=output, inputs=input_tensor, grad_outputs=gradient_output
        )[0]
        normalized_relevance = (relevance / torch.max(torch.abs(relevance))).sum(dim=1)

    return normalized_relevance, relevance
