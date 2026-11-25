import torch
import samplers
import cuda_sampling_v3

from sdllm import load_model
from sampling import sample

model, tokenizer, config = load_model(".", "cuda")

model.config.sampling.cuda_v3 = True
config.sampling.cuda_v3 = True

original = cuda_sampling_v3.sample_cuda_v3

stats = {
    "steps": 0,
    "tokens": 0,
    "matched": 0,
    "active": 0,
    "active_matched": 0
}

def verify(probs, x, mask_index, **kwargs):

    temperature = kwargs.get("temperature", 1.0)

    noise = torch.rand_like(
        probs, dtype=torch.float64
    )

    # PyTorch reference
    if temperature == 0.0:
        ref = probs.argmax(dim=-1)
    else:
        ref = (
            probs / (-torch.log(noise)) ** temperature
        ).argmax(dim=-1)

    active = x == mask_index
    ref = torch.where(active, ref, x)

    # CUDA V3: identical random noise
    kwargs["noise"] = noise

    out = original(
        probs, x, mask_index, **kwargs
    )

    matched = ref == out

    stats["steps"] += 1
    stats["tokens"] += x.numel()
    stats["matched"] += matched.sum().item()
    stats["active"] += active.sum().item()
    stats["active_matched"] += (
        matched & active
    ).sum().item()

    if not matched.all().item():
        raise AssertionError(
            f"Mismatch at step {stats['steps']}"
        )

    return out

cuda_sampling_v3.sample_cuda_v3 = verify

prompts = [
    "The capital of France is",
    "Explain how a computer works.",
    "Once upon a time,",
    "The advantages of renewable energy are",
    "Python is a programming language that"
]

with torch.inference_mode():
    for seed in [42, 1234, 5678]:
        for text in prompts:

            torch.manual_seed(seed)

            prompt = tokenizer.encode(
                text
            ).to(model.device)

            sample(
                model, prompt, 1, 128, 64, "none"
            )

            print(
                f"Seed={seed} | "
                f"Prompt={text[:30]} | PASS"
            )

assert stats["steps"] > 0
assert stats["active"] > 0

print("\n=== Final Validation ===")
print("Sampling calls:", stats["steps"])
print("Total tokens:", stats["tokens"])
print("Matched tokens:", stats["matched"])
print("Active tokens:", stats["active"])
print("Active matched:", stats["active_matched"])

print("Token Match:",
      stats["matched"] / stats["tokens"])

print("Active Token Match:",
      stats["active_matched"] / stats["active"])

print("RESULT: PASS")
