from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from models.clip_wrapper import CLIPWrapper

# ViT-B-32: 224 / 32 = 7 patches per side
_GRID_SIZE = 7


class GradCAM:
    """Grad-CAM for CLIP ViT-B-32 image encoder.

    Hooks into the last transformer block of the visual encoder to produce
    a spatial heatmap showing which image regions contribute most to the
    cosine similarity with a given ingredient text prompt.
    """

    def __init__(self, clip: CLIPWrapper) -> None:
        self.clip = clip
        self._activations: dict = {}
        self._gradients: dict = {}
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        last_block = self.clip.model.visual.transformer.resblocks[-1]

        def forward_hook(module, input, output):
            self._activations["value"] = output

        def backward_hook(module, grad_input, grad_output):
            self._gradients["value"] = grad_output[0].detach()

        self._handles.append(last_block.register_forward_hook(forward_hook))
        self._handles.append(last_block.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(
        self,
        image: torch.Tensor,
        ingredient_text: str,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for one image and one ingredient.

        Args:
            image: (1, 3, 224, 224) preprocessed image tensor.
            ingredient_text: ingredient string, e.g. "tomato".

        Returns:
            (224, 224) float32 numpy array with values in [0, 1].
        """
        image = image.to(self.clip.device)

        # encode text (no grad needed)
        text_emb = self.clip.encode_text([ingredient_text])  # (1, D)

        # encode image WITH grad so backward pass works
        self.clip.model.zero_grad()
        with torch.enable_grad():
            raw_emb = self.clip.model.encode_image(image)              # (1, D)
            image_emb = F.normalize(raw_emb, dim=-1)
            similarity = (image_emb * text_emb).sum()
            similarity.backward()

        # activations shape: (N_tokens, 1, D) where N_tokens = 1 + 7*7 = 50
        activations = self._activations["value"]  # (50, 1, D)
        gradients = self._gradients["value"]       # (50, 1, D)

        # drop CLS token, keep patch tokens → (49, 1, D)
        patch_act = activations[1:]   # (49, 1, D)
        patch_grad = gradients[1:]    # (49, 1, D)

        # weight activations channel-wise by gradients, then average
        cam = (patch_grad * patch_act).mean(dim=-1).squeeze(-1)  # (49,)

        # ReLU and reshape to spatial grid
        cam = F.relu(cam)
        cam = cam.reshape(_GRID_SIZE, _GRID_SIZE)  # (7, 7)

        # upsample to 224x224
        cam = cam.unsqueeze(0).unsqueeze(0)        # (1, 1, 7, 7)
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze()                                 # (224, 224)

        # normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)
