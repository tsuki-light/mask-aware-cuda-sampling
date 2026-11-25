"""
用真實 q_xs 量測 V3 的 skip_zero（跳過 p == 0）效果。
先跑 capture_qxs.py 產生 real_qxs.pt；這支本身不需要模型。

每個被抓下的步驟量三件事：
  1. 稀疏度：還在遮罩的列，每列幾個非零；
     含非零的 32 格區塊比例 = 需要跑 FP64 路徑的 warp 比例
  2. 正確性：shared noise 下，skip 關 / 開 都要跟 PyTorch 參考逐 token 相同
  3. 速度：native RNG，skip 關 vs 開，並跟「讀完 probs」的記憶體下限比

用法：
  python benchmark_sparse.py
  python benchmark_sparse.py --csv ../results/sparse_skip.csv
"""
import argparse
import csv

import torch
from cuda_sampling_v3 import sample_cuda_v3

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="real_qxs.pt")
parser.add_argument("--bandwidth", type=float, default=192.0,
                    help="GPU 記憶體頻寬 GB/s（RTX 4050 Laptop = 192）")
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--csv", default=None)
args = parser.parse_args()

data = torch.load(args.data, weights_only=True)
MASK = data["mask_index"]

torch.manual_seed(0)

def timed(fn):
    for i in range(10):
        fn(i)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(args.iters):
        fn(i + 10)

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / args.iters

header = (
    f"{'call':>4} | {'masked':>10} | {'nz/row':>7} | {'warp%':>6} | "
    f"{'off ms':>7} | {'on ms':>7} | {'speedup':>7} | "
    f"{'floor ms':>8} | {'on GB/s':>7} | match"
)
print(header)
print("-" * len(header))

rows_out = []
all_match = True

for step in data["steps"]:

    probs = step["probs"].to("cuda", torch.float32).unsqueeze(0).contiguous()
    x = step["x"].to("cuda", torch.int64).unsqueeze(0).contiguous()

    rows, vocab = probs.shape[1], probs.shape[2]
    active = (x[0] == MASK)
    n_active = int(active.sum())

    if n_active == 0:
        print(f"{step['call']:4d} | 沒有遮罩列，跳過")
        continue

    # 1. 稀疏度：只看還在遮罩的列（只有它們會被 kernel 算）
    nz = probs[0][active] != 0
    nz_per_row = nz.sum(-1).float().mean().item()

    usable = vocab // 32 * 32
    warp_frac = (
        nz[:, :usable]
        .reshape(n_active, -1, 32)
        .any(-1)
        .float()
        .mean()
        .item()
    )

    # 2. 正確性：同一份 FP64 noise
    noise = torch.rand_like(probs, dtype=torch.float64)

    ref = (probs / (-torch.log(noise))).argmax(dim=-1)
    ref = torch.where(x == MASK, ref, x)

    out_off = sample_cuda_v3(probs, x, MASK, noise=noise, skip_zero=False)
    out_on = sample_cuda_v3(probs, x, MASK, noise=noise, skip_zero=True)

    match = torch.equal(out_off, ref) and torch.equal(out_on, ref)
    all_match &= match

    del noise

    # 3. 速度：native RNG
    t_off = timed(lambda i: sample_cuda_v3(
        probs, x, MASK, nonce=i, skip_zero=False))
    t_on = timed(lambda i: sample_cuda_v3(
        probs, x, MASK, nonce=i, skip_zero=True))

    bytes_read = n_active * vocab * 4
    floor_ms = bytes_read / (args.bandwidth * 1e9) * 1e3
    on_gbs = bytes_read / (t_on * 1e-3) / 1e9

    print(
        f"{step['call']:4d} | {n_active:4d}/{rows:<5d} | "
        f"{nz_per_row:7.0f} | {warp_frac * 100:5.1f}% | "
        f"{t_off:7.3f} | {t_on:7.3f} | {t_off / t_on:6.2f}x | "
        f"{floor_ms:8.3f} | {on_gbs:7.1f} | {match}"
    )

    rows_out.append({
        "call": step["call"],
        "masked_rows": n_active,
        "total_rows": rows,
        "nonzero_per_row": round(nz_per_row, 1),
        "warp_fraction": round(warp_frac, 4),
        "skip_off_ms": round(t_off, 3),
        "skip_on_ms": round(t_on, 3),
        "speedup": round(t_off / t_on, 3),
        "memory_floor_ms": round(floor_ms, 3),
        "skip_on_gbs": round(on_gbs, 1),
        "match": match,
    })

if args.csv and rows_out:
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved {args.csv}")

assert rows_out, "沒有可量測的步驟"
assert all_match, "shared-noise 結果不一致"

print("\nRESULT: PASS (shared-noise 逐 token 相同)")
