import torch
import samplers

from sdllm import load_model
from sampling import sample

model, tokenizer, config = load_model(".", "cuda")

model.config.sampling.cuda_v3 = False
config.sampling.cuda_v3 = False

original = samplers.sample_categorical
captured = []

def capture(probs, *args, **kwargs):
    if probs.ndim == 3:
        captured.append(probs.detach().cpu())
    return original(probs, *args, **kwargs)

samplers.sample_categorical = capture

torch.manual_seed(42)

prompt = tokenizer.encode(
    "The capital of France is"
).to(model.device)

with torch.inference_mode():
    sample(model, prompt, 1, 128, 16, "none")

samplers.sample_categorical = original

rows = torch.cat([
    p.reshape(-1, p.shape[-1])
    for p in captured
], dim=0)

mask = model.mask_index
mass = rows.sum(dim=-1)
valid = (mass > 0) & torch.isfinite(rows).all(dim=-1)
rows = rows[valid]
rows = rows / rows.sum(dim=-1, keepdim=True)

# 排除完全由 MASK 或單一 token 主導的分布
confidence = rows.max(dim=-1).values
valid = confidence < 0.999
rows = rows[valid]

confidence = rows.max(dim=-1).values
order = confidence.argsort()

selected = {
    "low_confidence": rows[order[0]].clone(),
    "high_confidence": rows[order[-1]].clone(),
    "mask_competition": rows[
        (rows[:, mask] - 0.5).abs().argmin()
    ].clone()
}

torch.save({
    "probs": selected,
    "mask_index": mask
}, "real_probs.pt")

for name, p in selected.items():
    print(
        name,
        "| Vocab:", p.numel(),
        "| Max probability:", p.max().item(),
        "| Mask probability:", p[mask].item(),
        "| Nonzero:", (p > 0).sum().item()
    )

print("Saved real_probs.pt")
