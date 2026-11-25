import torch
from cuda_sampling_v2 import sample_cuda_v2
from cuda_sampling_v3 import sample_cuda_v3

torch.manual_seed(42)

B, N, V = 1, 133, 131072
MASK = V - 1

probs = torch.rand(
    (B, N, V),
    device="cuda",
    dtype=torch.float32
)

probs /= probs.sum(dim=-1, keepdim=True)

noise = torch.rand_like(probs, dtype=torch.float64)
reference = (probs / (-noise.log())).argmax(-1)

def benchmark(fn):
    for i in range(10):
        fn(i)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(100):
        fn(i + 10)

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / 100

v2 = benchmark(
    lambda i: sample_cuda_v2(probs, nonce=i)
)

print(f"V2 Baseline: {v2:.3f} ms")

for ratio in [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]:

    x = torch.zeros(
        (B, N), device="cuda", dtype=torch.int64
    )

    count = int(N * ratio)
    x[:, :count] = MASK

    expected = torch.where(
        x == MASK, reference, x
    )

    result = sample_cuda_v3(
        probs, x, MASK, noise=noise
    )

    assert torch.equal(expected, result)

    ms = benchmark(
        lambda i: sample_cuda_v3(
            probs, x, MASK, nonce=i
        )
    )

    print(
        f"Masked: {count:3d}/{N} | "
        f"V3: {ms:.3f} ms | "
        f"Speedup: {v2/ms:.3f}x | "
        f"Correct: True"
    )
