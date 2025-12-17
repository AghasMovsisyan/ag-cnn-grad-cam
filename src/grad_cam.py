import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt


GRADCAM_COUNTER = 1


def register_hooks_by_name(
    model: nn.Module, layer_name: str = "", save_all: bool = False
):
    activations = {}
    gradients = {}
    handles = []

    def save_forward(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                gradients[name] = output[0].detach()
            else:
                activations[name] = output.detach()

        return hook

    def save_backward(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().clone()

        return hook

    named_layers = list(model.named_modules())

    if save_all:
        for name, layer in named_layers:
            if name:
                handles.append(layer.register_forward_hook(save_forward(name)))
                handles.append(layer.register_full_backward_hook(save_backward(name)))
    else:
        exact_match = [
            (name, layer) for name, layer in named_layers if name == layer_name
        ]
        if exact_match:
            name, layer = exact_match[0]
        else:
            matching = [
                (name, layer)
                for name, layer in named_layers
                if name.startswith(layer_name)
            ]
            if not matching:
                all_names = [name for name, _ in named_layers if name]
                raise ValueError(
                    f"No layer found with name '{layer_name}' or starting with it.\n"
                    f"Available layer names:\n{', '.join(all_names)}"
                )
            if len(matching) > 1:
                available = [name for name, _ in matching]
                raise ValueError(
                    f"Multiple layers found starting with '{layer_name}': {available}. Please specify exact layer name."
                )
            name, layer = matching[0]
        handles.append(layer.register_forward_hook(save_forward(name)))
        handles.append(layer.register_full_backward_hook(save_backward(name)))

    return activations, gradients, handles


def cleanup_hooks(handles):
    for handle in handles:
        handle.remove()


def grad_cam(
    model,
    image: torch.Tensor,
    layer_name: str = None,
    save_all: bool = False,
    output_dir: str = "cam_outputs",
    print_shapes: bool = True,
    is_lstm_model: bool = False,
):
    """
    Grad-CAM computation with sequential folder saving:
    Each CAM + original + heatmap_plot saved in a separate folder
    """
    global GRADCAM_COUNTER

    if save_all and layer_name is not None:
        raise ValueError(
            "Cannot use `save_all=True` and `layer_name` at the same time."
        )
    if not save_all and layer_name is None:
        raise ValueError("If `save_all=False`, you must specify `layer_name`.")

    if is_lstm_model:
        model.train()
    else:
        model.eval()

    activations, gradients, handles = register_hooks_by_name(
        model, layer_name, save_all
    )
    logits = model(image)
    predicted_class_idx = logits.argmax(dim=1).item()
    model.zero_grad()
    logits[0, predicted_class_idx].backward()

    def compute_and_save_cam(
        activation: torch.Tensor,
        gradient: torch.Tensor,
        name: str,
        image: Optional[torch.Tensor] = None,
        base_save_dir: str = None,
        print_shapes: bool = False,
    ):
        global GRADCAM_COUNTER

        folder_name = os.path.join(base_save_dir, f"cam_{GRADCAM_COUNTER}")
        os.makedirs(folder_name, exist_ok=True)

        # if print_shapes:
        #     print(
        #         f"[{name}] Activation: {activation.shape}, Gradient: {gradient.shape}"
        #     )

        if len(activation.shape) != 4 or len(gradient.shape) != 4:
            print(f"Skipping layer {name} due to incompatible shape (not 4D).")
            return

        if name in ["global_avg_pool", "gap", "avgpool", "adaptive_avg_pool2d"]:
            print(f"Skipping layer {name} - pooling layers destroy spatial info.")
            return

        if activation.shape[2] <= 2 or activation.shape[3] <= 2:
            print(
                f"Skipping layer {name} - spatial dims too small ({activation.shape[2]}x{activation.shape[3]})."
            )
            return

        pooled_grad = torch.mean(gradient, dim=[0, 2, 3])
        weighted_activation = activation * pooled_grad.view(1, -1, 1, 1)

        heatmap = torch.mean(weighted_activation, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap).clamp(min=1e-6)
        heatmap_np = heatmap.cpu().numpy()

        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_TURBO)

        if image is not None:
            image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image_np = (image_np - image_np.min()) / (
                image_np.max() - image_np.min() + 1e-6
            )
            image_np = np.uint8(255 * image_np)
            heatmap_img = cv2.resize(
                heatmap_img, (image_np.shape[1], image_np.shape[0])
            )
            superimposed = cv2.addWeighted(image_np, 0.4, heatmap_img, 0.6, 0)

            cv2.imwrite(
                os.path.join(folder_name, f"gradcam_{GRADCAM_COUNTER}.jpg"),
                superimposed,
            )
            plt.imsave(
                os.path.join(folder_name, f"original_{GRADCAM_COUNTER}.jpg"), image_np
            )
            plt.imsave(
                os.path.join(folder_name, f"heatmap_plot_{GRADCAM_COUNTER}.jpg"),
                heatmap_np,
            )

            GRADCAM_COUNTER += 1

    if save_all:
        for name in activations:
            activation = activations[name]
            gradient = gradients.get(name)
            if gradient is not None:
                compute_and_save_cam(
                    activation,
                    gradient,
                    name,
                    image,
                    base_save_dir=output_dir,
                    print_shapes=print_shapes,
                )
            else:
                print(f"Skipping layer {name} - no gradient available.")
    else:
        activation = activations[layer_name]
        gradient = gradients.get(layer_name)
        if gradient is not None:
            compute_and_save_cam(
                activation,
                gradient,
                layer_name,
                image,
                base_save_dir=output_dir,
                print_shapes=print_shapes,
            )
        else:
            print(f"No gradient available for layer {layer_name}")

    cleanup_hooks(handles)
