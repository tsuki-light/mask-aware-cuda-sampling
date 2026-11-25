import time
import statistics
import torch
import gc
import cuda_sampling_v3

from sdllm import load_model
from sampling import sample

original = cuda_sampling_v3.sample_cuda_v3
skip_zero = False

def dispatch(*args, **kwargs):
    kwargs["skip_zero"] = skip_zero
    return original(*args, **kwargs)

cuda_sampling_v3.sample_cuda_v3 = dispatch

model, tokenizer, config = load_model(".", "cuda")

model.config.sampling.cuda_v3 = True
config.sampling.cuda_v3 = True

prompt = tokenizer.encode(
    "The capital of France is"
).to(model.device)

RUNS = 10

@torch.inference_mode()
def run(enabled, length, steps):
    global skip_zero
    skip_zero = enabled

    model.config.model.length = length
    model.num_tokens = length

    torch.manual_seed(42)

    return sample(
        model, prompt, 1, length, steps, "none"
    )

for length in [64, 128]:
    for steps in [16, 32, 64]:

        results = {False: [], True: []}
        peaks = {False: [], True: []}

        print(f"\nLength={length}, Steps={steps}")

        for enabled in [False, True]:
            run(enabled, length, steps)
            torch.cuda.synchronize()

        for i in range(RUNS):
            order = [False, True] if i % 2 == 0 else [True, False]

            for enabled in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()
                run(enabled, length, steps)
                torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated() / 1024**3

                results[enabled].append(elapsed)
                peaks[enabled].append(peak)

        a = statistics.median(results[False])
        b = statistics.median(results[True])

        print(f"CUDA V3: {a:.3f}s")
        print(f"CUDA V5: {b:.3f}s")
        print(f"V5 Speedup: {a/b:.3f}x")
        print(f"V3 VRAM: {max(peaks[False]):.3f} GiB")
        print(f"V5 VRAM: {max(peaks[True]):.3f} GiB")

        wins = sum(
            b < a for a, b in
            zip(results[False], results[True])
        )

        print(f"V5 Wins: {wins}/{RUNS}")

        gc.collect()
        torch.cuda.empty_cache()
