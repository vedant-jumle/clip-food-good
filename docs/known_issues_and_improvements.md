# Known Issues and Potential Improvements

---

## Visualization — GradCAM is ineffective on ViT

**Status:** Known issue, not yet fixed.

### What's wrong
The current `src/visualization/gradcam.py` produces near-identical, mostly dark heatmaps across all ingredients for a given image. Two symptoms:
1. Every ingredient prediction for the same image produces the same spatial activation pattern — the gradients don't differentiate between ingredients.
2. Heatmaps are almost entirely dark (near-zero) after ReLU — the weighted activations `(patch_grad * patch_act)` have a negative mean, so ReLU kills most signal.

### Root cause
Grad-CAM was designed for CNNs with spatial feature maps. ViT (which CLIP uses) aggregates spatial information into a CLS token in the final block — by that point, patch gradients are spatially uniform and don't localize to specific image regions. Hooking the last resblock gives meaningless gradients.

### Recommended fix: Attention Rollout
Replace `GradCAM` with `AttentionRollout` — the standard technique for ViT visualization:
- Collect attention weight matrices from every transformer block (averaged across heads)
- Multiply them recursively across layers (rollout), adding a residual identity at each step
- The result is a `(49,)` token-importance map that can be upsampled to `(224, 224)`

This requires no gradient computation — hooks capture `attn_weights` during the forward pass only. Same `__call__(image, ingredient)` interface, so `run_visualization.py` needs minimal changes.

Alternative quick attempt: hook an earlier block (e.g. block 8 or 9) instead of the last one — earlier blocks retain more spatial locality, though results are still likely weaker than rollout.

---

## Prompt Engineering — Ensemble diluted by weak prompt A

**Status:** Observed, partially addressed.

The 4-prompt ensemble (A+B+C+D) matches but doesn't beat prompt B alone (F1@5=0.11 for both). Prompt A (single word) is the weakest and drags down the average. The ENS7 variant (B+C+D+E+F+G+H, no A) is implemented in `run_experiment2.py` but results are pending.

---

## Fine-tuning — Underperforms zero-shot at current data scale

**Status:** Expected, documented finding.

With ~1,880 training samples, both projection head and partial unfreeze degrade performance vs zero-shot. This is consistent with the literature — CLIP fine-tuning typically requires tens of thousands of samples. Fix: download more Recipe1M+ shards. With the full ~1M recipes, fine-tuning results would likely reverse.

---

## Adaptive threshold — Saturates at max_k

**Status:** Observed.

`predict_adaptive` selected k=10 (the configured max) on the test set. This means CLIP's score distributions are flat — there's no strong separation between predicted and non-predicted ingredients, so many tokens exceed the `mean + 0.5*std` threshold. Options:
- Increase `alpha` (e.g. 1.0 or 1.5) to raise the threshold and reduce k
- Tune `max_k` based on average ground-truth ingredient count in the dataset