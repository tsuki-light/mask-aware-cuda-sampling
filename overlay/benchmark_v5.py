import runpy
import cuda_sampling_v3

original = cuda_sampling_v3.sample_cuda_v3

def sample_v5(*args, **kwargs):
    kwargs["skip_zero"] = True
    return original(*args, **kwargs)

cuda_sampling_v3.sample_cuda_v3 = sample_v5

runpy.run_path("benchmark_final.py", run_name="__main__")
