import time
import torch
import samplers

from sdllm import load_model
from sampling import sample
from cuda_sampling import sample_cuda

original = samplers.sample_categorical

model, tokenizer, config = load_model(".", "cuda")

prompt = tokenizer.encode(
    "The capital of France is"
).to(model.device)

nonce = 0
calls = 0
fallbacks = 0

def custom(probs, temperature=1.0, dp=False):
    global nonce, calls, fallbacks

    if (
        not probs.is_cuda
        or probs.dtype != torch.float32
        or not probs.is_contiguous()
        or temperature <= 0
        or not dp
    ):
        fallbacks += 1
        return original(probs, temperature, dp)

    if calls == 0:
        print("Actual input:", probs.shape, probs.dtype)

    calls += 1
    nonce += 1

    return sample_cuda(
        probs,
        seed=42,
        nonce=nonce,
        temperature=temperature
    )

@torch.inference_mode()
def run():
    global nonce
    nonce = 0
    torch.manual_seed(42)
    return sample(model, prompt, 1, 32, 16, "none")

def benchmark(name, fn):
    global calls, fallbacks

    samplers.sample_categorical = fn

    run()
    torch.cuda.synchronize()

    calls = 0
    fallbacks = 0

    times = []
    peaks = []

    for i in range(5):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        output = run()
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3

        times.append(elapsed)
        peaks.append(peak)

        print(f"{name} {i+1}: {elapsed:.3f}s | {peak:.3f} GiB")

    avg = sum(times) / len(times)

    print(f"{name} Average: {avg:.3f}s")
    print(f"{name} Peak VRAM: {max(peaks):.3f} GiB")
    print("CUDA calls:", calls)
    print("Fallbacks:", fallbacks)
    print("Output:", tokenizer.decode(output[0, 5:37]))

    return avg

baseline = benchmark("PyTorch", original)
optimized = benchmark("Custom CUDA", custom)

print(f"End-to-End Speedup: {baseline / optimized:.3f}x")
