import torch
import numpy as np
from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon

from samplers import sample_categorical
from cuda_sampling_v3 import sample_cuda_v3

torch.manual_seed(42)

N = 4096
RUNS = 256
MASK = 7

expected = np.array([
    0.35, 0.25, 0.15, 0.10,
    0.07, 0.04, 0.03, 0.01
])

V = len(expected)

probs = torch.tensor(
    expected, device="cuda", dtype=torch.float32
).repeat(1, N, 1)

x = torch.full(
    (1, N), MASK,
    device="cuda", dtype=torch.int64
)

counts = {
    "PyTorch": torch.zeros(V, device="cuda", dtype=torch.int64),
    "CUDA V3": torch.zeros(V, device="cuda", dtype=torch.int64)
}

for i in range(RUNS):

    a = sample_categorical(
        probs, temperature=1.0, dp=True
    )

    b = sample_cuda_v3(
        probs, x, MASK,
        seed=42, nonce=i
    )

    counts["PyTorch"] += torch.bincount(
        a.flatten(), minlength=V
    )

    counts["CUDA V3"] += torch.bincount(
        b.flatten(), minlength=V
    )

torch.cuda.synchronize()

total = N * RUNS

a = counts["PyTorch"].cpu().numpy()
b = counts["CUDA V3"].cpu().numpy()

pa = a / total
pb = b / total

print(f"\nSamples per implementation: {total:,}")

print("\nToken | Expected | PyTorch | CUDA V3")
for i in range(V):
    print(
        f"{i:5d} | {expected[i]:.6f} | "
        f"{pa[i]:.6f} | {pb[i]:.6f}"
    )

print("\n--- Distribution Validation ---")

print("PyTorch TVD:",
      0.5 * np.abs(pa - expected).sum())

print("CUDA V3 TVD:",
      0.5 * np.abs(pb - expected).sum())

print("PyTorch vs CUDA TVD:",
      0.5 * np.abs(pa - pb).sum())

print("JS Divergence:",
      jensenshannon(pa, pb, base=2) ** 2)

print("PyTorch Chi-square p-value:",
      chisquare(a, expected * total).pvalue)

print("CUDA V3 Chi-square p-value:",
      chisquare(b, expected * total).pvalue)
