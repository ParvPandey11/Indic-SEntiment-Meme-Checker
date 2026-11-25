"""Grad-CAM utilities for the final ResNet-50 convolution block."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def gradcam_overlay(model, image_tensor, enc=None, target_class=None, alpha=0.42):
    """Return (original PIL image, overlay PIL image, predicted class, logits).

    `image_tensor` must be a normalized 1x3x224x224 tensor. For fusion models,
    `enc` is the tokenized text input. The class score is differentiated through
    the model while capturing activations and gradients from ResNet layer4.
    """
    if not hasattr(model, "img_enc"):
        raise ValueError("Grad-CAM requires an image or fusion model")
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("Expected one image tensor with shape 1x3xHxW")

    captured = {}

    def capture_activation(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradient", grad))

    handle = model.img_enc.layer4[-1].register_forward_hook(capture_activation)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(image_tensor, enc)
            pred = int(logits.argmax(dim=1).item())
            class_id = pred if target_class is None else int(target_class)
            logits[0, class_id].backward()
        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        lo, hi = float(cam.min()), float(cam.max())
        cam = (cam - lo) / (hi - lo + 1e-8)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_array = (image_tensor.detach().cpu()[0] * std + mean).clamp(0, 1)
        original_array = (original_array.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        original = Image.fromarray(original_array, mode="RGB")

        # Use Matplotlib's built-in color map; it is already a core dependency.
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import colormaps
        heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
        overlay = Image.blend(original, Image.fromarray(heat, mode="RGB"), alpha=alpha)
        return original, overlay, pred, logits.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
