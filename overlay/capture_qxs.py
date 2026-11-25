"""
抓真實的 q_xs（整個送進 sampler 的機率張量）和當下的 mask 狀態，
給 benchmark_sparse.py 用。

需要本地 MDLM 整合（跟 test_e2e_equivalence.py 一樣，
攔截 cuda_sampling_v3.sample_cuda_v3）。

輸出 real_qxs.pt：
  steps: list of {"call": int, "probs": (rows, vocab) fp32, "x": (rows,) int64}
  mask_index: int
  total_calls: int
"""
import torch
import cuda_sampling_v3

from sdllm import load_model
from sampling import sample

PROMPT = "The capital of France is"
LENGTH = 128
STEPS = 64
KEEP = 5   # 平均挑幾個步驟存下來；每個約 rows × 131,072 × 4 B ≈ 70 MB

model, tokenizer, config = load_model(".", "cuda")

model.config.sampling.cuda_v3 = True
config.sampling.cuda_v3 = True

original = cuda_sampling_v3.sample_cuda_v3

keep = {
    round(k * (STEPS - 1) / (KEEP - 1))
    for k in range(KEEP)
}

state = {"calls": 0, "mask_index": None}
captured = []

def capture(probs, x, mask_index, **kwargs):

    if state["calls"] in keep:
        vocab = probs.shape[-1]
        captured.append({
            "call": state["calls"],
            "probs": probs.detach().reshape(-1, vocab).float().cpu(),
            "x": x.detach().reshape(-1).cpu(),
        })

    state["calls"] += 1
    state["mask_index"] = int(mask_index)

    return original(probs, x, mask_index, **kwargs)

cuda_sampling_v3.sample_cuda_v3 = capture

torch.manual_seed(42)

prompt = tokenizer.encode(PROMPT).to(model.device)

with torch.inference_mode():
    sample(model, prompt, 1, LENGTH, STEPS, "none")

cuda_sampling_v3.sample_cuda_v3 = original

assert state["calls"] > 0, (
    "沒攔到 sample_cuda_v3：確認整合是透過 "
    "cuda_sampling_v3.sample_cuda_v3(...) 呼叫"
)

torch.save({
    "steps": captured,
    "mask_index": state["mask_index"],
    "total_calls": state["calls"],
}, "real_qxs.pt")

print(f"Sampling calls: {state['calls']}, captured: {len(captured)}")

for s in captured:
    active = s["x"] == state["mask_index"]
    nz = (s["probs"][active] != 0).sum(-1).float()
    print(
        f"call {s['call']:3d} | rows {s['x'].numel()} | "
        f"masked {int(active.sum()):3d} | "
        f"nonzero/row mean {nz.mean().item() if nz.numel() else 0:.0f}"
    )

print("Saved real_qxs.pt")
