import torch
from .inference import forward_pass


def make_gradient_output(output, predicted_class):
    gradient_output = torch.zeros_like(output).to(output.device)
    predicted_class = torch.as_tensor(predicted_class, device=output.device)
    if predicted_class.ndim == 0:
        predicted_class = predicted_class.expand(output.shape[0])

    batch_index = torch.arange(output.shape[0], device=output.device)
    gradient_output[batch_index, predicted_class] = output[batch_index, predicted_class]
    return gradient_output


# def explain_model_prediction(model, input_tensor):

#     if input_tensor.grad is not None:
#         input_tensor.grad.zero_()
#     input_tensor.requires_grad = True
#     output = forward_pass(model, input_tensor, gradient_required=True)

#     predicted_class = output.argmax(dim=1)

#     gradient_output = make_gradient_output(output, predicted_class)

#     # now compute gradients with torch.autograd
#     with torch.no_grad():
#         relevance = torch.autograd.grad(
#             outputs=output, inputs=input_tensor, grad_outputs=gradient_output
#         )[0]
#         normalized_relevance = (relevance / torch.max(torch.abs(relevance))).sum(dim=1)

#     return normalized_relevance, relevance


# def explain_output(model, input_tensor, output_index):

#     if input_tensor.grad is not None:
#         input_tensor.grad.zero_()
#     input_tensor.requires_grad = True
#     output = forward_pass(model, input_tensor, gradient_required=True)

#     if output_index is None:
#         predicted_class = output.argmax(dim=1)
#         gradient_output = make_gradient_output(output, predicted_class)
#     elif output_index == "logit":
#         gradient_output = output
#     elif output_index == "ones":
#         gradient_output = torch.ones_like(output).to(output.device)
#     else:
#         predicted_class = output_index
#         gradient_output = make_gradient_output(output, predicted_class)

#     # now compute gradients with torch.autograd
#     with torch.no_grad():
#         relevance = torch.autograd.grad(
#             outputs=output, inputs=input_tensor, grad_outputs=gradient_output
#         )[0]
#         normalized_relevance = (relevance / torch.max(torch.abs(relevance))).sum(dim=1)

#     return normalized_relevance, relevance


def compute_relevance(
    model,
    input_tensor,
    targeted=False,
    target=None,
    init_relevance="logit",
    normalize=False,
    tuple_output_index=0,
):

    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    input_tensor.requires_grad = True

    output = forward_pass(model, input_tensor, gradient_required=True)
    # if output is a tuple, take the given element out
    # if not isinstance(output, tuple):
    #     output = output[tuple_output_index]

    if targeted:
        if target is None:
            # first check if the shape of output is compatible with argmax
            # targeted_index = output.argmax(dim=1)
            if output.ndim == 1:
                targeted_index = output.argmax()
            elif output.ndim == 2:
                targeted_index = output.argmax(dim=1)
            gradient_output = make_gradient_output(output, targeted_index)
        else:
            targeted_index = target
            gradient_output = make_gradient_output(output, targeted_index)

        if init_relevance == "logit":
            pass
        elif init_relevance == 1:
            gradient_output = make_gradient_output(
                torch.ones_like(output), targeted_index
            )
        elif init_relevance == -1:
            gradient_output = make_gradient_output(
                -torch.ones_like(output), targeted_index
            )

        output.backward(gradient_output)

    else:
        output.backward()

    relevance = input_tensor.grad.detach() * input_tensor.detach()

    if normalize:
        relevance = relevance / torch.max(torch.abs(relevance))

    return relevance, output.detach()
