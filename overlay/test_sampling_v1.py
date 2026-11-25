import torch
from cuda_sampling import sample_cuda
from samplers import sample_categorical

B, N, V = 1, 133, 131072

torch.manual_seed(42)

probs = torch.rand(
    (B, N, V),
    device="cuda",
    dtype=torch.float32
)

probs /= probs.sum(dim=-1, keepdim=True)

# Correctness: identical random numbers
noise = torch.rand_like(probs, dtype=torch.float64)

reference = (probs / (-torch.log(noise))).argmax(-1)
result = sample_cuda(probs, noise=noise)

torch.cuda.synchronize()

correct = torch.equal(reference, result)

print("Correct:", correct)
print("Matched:", (reference == result).sum().item(), "/", B*N)

assert correct, "Correctness test failed"

del noise, reference, result

# Benchmark
def benchmark(name, fn):
    for i in range(10):
        fn(i)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(100):
        fn(i + 10)

    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / 100
    peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"{name}: {ms:.3f} ms | VRAM: {peak:.3f} MiB")
    return ms

baseline = benchmark(
    "PyTorch",
    lambda i: sample_categorical(probs, dp=True)
)

optimized = benchmark(
    "CUDA",
    lambda i: sample_cuda(probs, nonce=i)
)

print(f"Speedup: {baseline / optimized:.3f}x")
