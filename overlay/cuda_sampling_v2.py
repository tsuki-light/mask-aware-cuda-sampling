import ctypes
import torch
from pathlib import Path

lib = ctypes.CDLL(
    str(Path(__file__).parent / "build/libsampling_v2.so")
)

fn = lib.sampling_forward_v2

fn.argtypes = (
    [ctypes.c_void_p] * 5 +
    [ctypes.c_int] * 2 +
    [ctypes.c_uint64] * 2 +
    [ctypes.c_double, ctypes.c_void_p]
)

fn.restype = ctypes.c_int

def sample_cuda_v2(probs, noise=None, seed=42,
                   nonce=0, temperature=1.0):

    assert probs.is_cuda
    assert probs.dtype == torch.float32
    assert probs.is_contiguous()
    assert temperature > 0

    if noise is not None:
        assert noise.device == probs.device
        assert noise.dtype == torch.float64
        assert noise.shape == probs.shape
        assert noise.is_contiguous()

    vocab = probs.shape[-1]
    rows = probs.numel() // vocab

    output = torch.empty(
        probs.shape[:-1],
        device=probs.device,
        dtype=torch.int64
    )

    values = torch.empty(
        (rows, 8), device=probs.device,
        dtype=torch.float64
    )

    indices = torch.empty(
        (rows, 8), device=probs.device,
        dtype=torch.int32
    )

    with torch.cuda.device(probs.device):
        stream = torch.cuda.current_stream()

        status = fn(
            probs.data_ptr(),
            noise.data_ptr() if noise is not None else None,
            output.data_ptr(),
            values.data_ptr(),
            indices.data_ptr(),
            rows, vocab, seed, nonce,
            temperature, stream.cuda_stream
        )

    if status != 0:
        raise RuntimeError(f"CUDA error: {status}")

    return output
