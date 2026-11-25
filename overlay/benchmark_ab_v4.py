import time
import torch
from sdllm import load_model
from sampling import sample

model, tokenizer, config = load_model(".", "cuda")
prompt = tokenizer.encode(
    "The capital of France is"
).to(model.device)

@torch.inference_mode()
def run(enabled):
    config.sampling.cuda_v4 = enabled
    model.config.sampling.cuda_v4 = enabled

    torch.manual_seed(42)

    return sample(model, prompt, 1, 32, 16, "none")

for enabled in [False, True]:
    run(enabled)
    torch.cuda.synchronize()

results = {False: [], True: []}

for i in range(10):
    for enabled in [False, True] if i % 2 == 0 else [True, False]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        output = run(enabled)
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3

        results[enabled].append(elapsed)

        name = "CUDA V4" if enabled else "PyTorch"
        print(f"{name}: {elapsed:.3f}s | {peak:.3f} GiB")

baseline = sum(results[False]) / 10
optimized = sum(results[True]) / 10

print(f"PyTorch: {baseline:.3f}s")
print(f"CUDA V4: {optimized:.3f}s")
print(f"Speedup: {baseline / optimized:.3f}x")
