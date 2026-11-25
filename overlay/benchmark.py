import time
import torch
from sdllm import load_model
from sampling import sample

model, tokenizer, config = load_model(".", "cuda")

prompt = tokenizer.encode(
    "The capital of France is"
).to(model.device)

@torch.inference_mode()
def run():
    torch.manual_seed(42)
    return sample(model, prompt, 1, 32, 16, "none")

print("Warming up...")
run()
torch.cuda.synchronize()

times = []

for i in range(3):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    run()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    times.append(elapsed)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Run {i+1}: {elapsed:.3f}s | Peak VRAM: {peak:.3f} GiB")

avg = sum(times) / len(times)

print(f"Average latency: {avg:.3f}s")
print(f"Steps/s: {16 / avg:.3f}")
