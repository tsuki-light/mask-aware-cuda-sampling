import torch
import numpy as np

from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon
from samplers import sample_categorical
from cuda_sampling_v3 import sample_cuda_v3

data = torch.load("real_probs.pt", weights_only=True)
MASK = data["mask_index"]

N = 128
RUNS = 2048

torch.manual_seed(1234)

for name, expected in data["probs"].items():
    if name != "low_confidence": continue

    expected = expected.double().numpy()
    expected /= expected.sum()
    V = len(expected)

    probs = torch.tensor(
        expected, device="cuda", dtype=torch.float32
    ).repeat(1, N, 1)

    x = torch.full(
        (1, N), MASK,
        device="cuda", dtype=torch.int64
    )

    ca = torch.zeros(V, device="cuda", dtype=torch.int64)
    cb = torch.zeros_like(ca)

    for i in range(RUNS):

        a = sample_categorical(
            probs, temperature=1.0, dp=True
        )

        b = sample_cuda_v3(
            probs, x, MASK, seed=5678, nonce=i
        )

        ca += torch.bincount(a.flatten(), minlength=V)
        cb += torch.bincount(b.flatten(), minlength=V)

    a = ca.cpu().numpy()
    b = cb.cpu().numpy()

    total = N * RUNS

    # 低頻 token 合併，避免卡方檢定失效
    common = expected * total >= 5

    def group(values):
        return np.append(
            values[common], values[~common].sum()
        )

    e = group(expected * total)
    ga, gb = group(a), group(b)
    valid = e > 0

    pa = a / total
    pb = b / total

    print(f"\n=== {name} ===")
    print("Samples:", total)
    print("Vocabulary:", V)
    print("PyTorch TVD:", 0.5 * abs(pa-expected).sum())
    print("CUDA V3 TVD:", 0.5 * abs(pb-expected).sum())
    print("Between TVD:", 0.5 * abs(pa-pb).sum())
    print("JS Divergence:", jensenshannon(pa,pb,base=2)**2)

    print("PyTorch p-value:",
          chisquare(ga[valid], e[valid]).pvalue)
    print("CUDA V3 p-value:",
          chisquare(gb[valid], e[valid]).pvalue)

    print("PyTorch impossible tokens:", a[expected==0].sum())
    print("CUDA impossible tokens:", b[expected==0].sum())
