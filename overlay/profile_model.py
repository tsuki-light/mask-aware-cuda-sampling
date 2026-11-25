import torch
from torch.profiler import profile, ProfilerActivity
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

print("Profiling...")

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
) as prof:
    run()
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=30
))

prof.export_chrome_trace("trace.json")
