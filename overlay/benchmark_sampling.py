import torch
from samplers import sample_categorical

B, N, V = 1, 128, 131072
WARMUP = 10
RUNS = 100

torch.manual_seed(42)

probs = torch.rand(
    (B, N, V),
    device="cuda",
    dtype=torch.float32
)

probs /= probs.sum(dim=-1, keepdim=True)

@torch.inference_mode()
def run():
    return sample_categorical(
        probs,
        temperature=1.0,
        dp=True
    )

for _ in range(WARMUP):
    run()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(RUNS):
    output = run()

end.record()
torch.cuda.synchronize()

print("Input shape:", probs.shape)
print("Input dtype:", probs.dtype)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
print("Latency:", start.elapsed_time(end) / RUNS, "ms")
print("Peak VRAM:", torch.cuda.max_memory_allocated() / 1024**2, "MiB")
