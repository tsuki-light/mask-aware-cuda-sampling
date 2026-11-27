import csv
import gc
import time
import statistics
import argparse
import torch

import cuda_sampling_v3

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--lengths", nargs="+", type=int,
                    default=[64, 128, 192, 256, 384, 512, 768, 1024])
args = parser.parse_args()

MODES = ["PyTorch", "V3", "V5"]
mode = "PyTorch"
calls = 0

original = cuda_sampling_v3.sample_cuda_v3


def dispatch(*a, **kw):
    global calls
    calls += 1
    kw["skip_zero"] = mode == "V5"
    return original(*a, **kw)


# Patch before importing the model
cuda_sampling_v3.sample_cuda_v3 = dispatch

from sdllm import load_model
from sampling import sample

model, tokenizer, config = load_model(".", "cuda")

prompt = tokenizer.encode(
    "The capital of France is"
).to(model.device)

prompt_length = prompt.numel()

print("GPU:", torch.cuda.get_device_name())
print("Prompt tokens:", prompt_length)
print("Backbone:", model.config.algo.backbone)


@torch.inference_mode()
def run(selected, length):
    global mode, calls

    mode = selected
    calls = 0

    model.config.sampling.cuda_v3 = mode != "PyTorch"
    model.config.model.length = length
    model.num_tokens = length

    torch.manual_seed(42)

    output = sample(
        model, prompt, 1,
        length - prompt_length,
        args.steps, "none"
    )

    if mode != "PyTorch" and calls == 0:
        raise RuntimeError("CUDA dispatch was not invoked")

    if mode == "PyTorch" and calls != 0:
        raise RuntimeError("Baseline invoked CUDA dispatch")

    return output


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


fields = [
    "mode", "canvas_length", "prompt_tokens",
    "generated_positions", "steps", "status",
    "runs", "p50_ms", "p95_ms",
    "effective_tps", "ms_per_token",
    "peak_allocated_mib", "peak_reserved_mib",
    "speedup", "latency_reduction_pct",
    "cuda_calls"
]

blocked = set()

with open("benchmark_extended.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for length in args.lengths:

        print(f"\n{'=' * 60}")
        print(f"Canvas={length}, Steps={args.steps}")
        print("=" * 60)

        results = {}

        for selected in MODES:

            if selected in blocked:
                continue

            cleanup()

            try:
                # Warmup
                output = run(selected, length)
                torch.cuda.synchronize()
                del output

                results[selected] = {
                    "times": [],
                    "allocated": [],
                    "reserved": [],
                    "tokens": [],
                    "calls": []
                }

            except torch.cuda.OutOfMemoryError:
                print(f"{selected}: OOM during warmup")
                blocked.add(selected)

                writer.writerow({
                    "mode": selected,
                    "canvas_length": length,
                    "steps": args.steps,
                    "status": "OOM"
                })
                f.flush()

            cleanup()

        for i in range(args.runs):

            order = MODES[i % 3:] + MODES[:i % 3]

            for selected in order:

                if selected not in results:
                    continue

                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                    start = time.perf_counter()

                    output = run(selected, length)

                    torch.cuda.synchronize()

                    elapsed = time.perf_counter() - start

                    tokens = output.shape[-1] - prompt_length

                    r = results[selected]
                    r["times"].append(elapsed)
                    r["tokens"].append(tokens)
                    r["calls"].append(calls)

                    r["allocated"].append(
                        torch.cuda.max_memory_allocated() / 2**20
                    )

                    r["reserved"].append(
                        torch.cuda.max_memory_reserved() / 2**20
                    )

                    del output

                    print(
                        f"{selected:8s} "
                        f"Run {i+1:2d}/{args.runs} "
                        f"{elapsed:.3f}s"
                    )

                except torch.cuda.OutOfMemoryError:
                    print(f"{selected}: OOM")
                    blocked.add(selected)
                    results.pop(selected)

                    writer.writerow({
                        "mode": selected,
                        "canvas_length": length,
                        "steps": args.steps,
                        "status": "OOM"
                    })
                    f.flush()

                    cleanup()

        baseline = None

        if "PyTorch" in results:
            baseline = statistics.median(
                results["PyTorch"]["times"]
            )

        for selected, r in results.items():

            times = r["times"]
            p50 = statistics.median(times)
            tokens = statistics.median(r["tokens"])

            p95 = (
                statistics.quantiles(
                    times, n=100, method="inclusive"
                )[94]
                if len(times) > 1 else None
            )

            speedup = baseline / p50 if baseline else None

            row = {
                "mode": selected,
                "canvas_length": length,
                "prompt_tokens": prompt_length,
                "generated_positions": tokens,
                "steps": args.steps,
                "status": "OK",
                "runs": len(times),
                "p50_ms": p50 * 1000,
                "p95_ms": p95 * 1000 if p95 else None,
                "effective_tps": tokens / p50,
                "ms_per_token": p50 * 1000 / tokens,
                "peak_allocated_mib": max(r["allocated"]),
                "peak_reserved_mib": max(r["reserved"]),
                "speedup": speedup,
                "latency_reduction_pct":
                    (1 - p50 / baseline) * 100
                    if baseline else None,
                "cuda_calls": statistics.median(r["calls"])
            }

            writer.writerow(row)
            f.flush()

            print(
                f"{selected:8s} | "
                f"P50={p50:.3f}s | "
                f"TPS={tokens/p50:.2f} | "
                f"VRAM={max(r['allocated']):.1f} MiB | "
                f"CUDA Calls={row['cuda_calls']}"
            )

        cleanup()

        if len(blocked) == len(MODES):
            break

print("\nSaved: benchmark_extended.csv")

