import torch


def forward_pass(model, input_tensor, gradient_required=False):
    # check device compatibility
    if model.device != input_tensor.device:
        input_tensor = input_tensor.to(model.device)

    if gradient_required:
        # clear previous gradients if exists
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        input_tensor.requires_grad = True
        output = model(input_tensor)
        return output
    else:
        with torch.no_grad():
            output = model(input_tensor)
    return output


def inference(model, input_tensor, class_names):
    output = forward_pass(model, input_tensor)
    predicted_class = output.argmax(dim=1).item()
    print(
        f"Predicted Class: {class_names[predicted_class]}, (Index: {predicted_class})"
    )
    return output
