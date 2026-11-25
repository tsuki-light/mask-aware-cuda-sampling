import ctypes
import torch
from pathlib import Path

lib = ctypes.CDLL(
    str(Path(__file__).parent / "build/libsampling_v3.so")
)

fn = lib.sampling_forward_v3

fn.argtypes = (
    [ctypes.c_void_p] * 6 +
    [ctypes.c_int64] +
    [ctypes.c_int] * 2 +
    [ctypes.c_uint64] * 2 +
    [ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
)

fn.restype = ctypes.c_int

def sample_cuda_v3(probs, x, mask_index, noise=None, seed=42,
                   nonce=0, temperature=1.0, skip_zero=False):
    """
    skip_zero=True: 跳過 p == 0 的詞表格子（top-p 之後大多是 0）。
      shared noise 下結果與 PyTorch 參考逐 token 相同；
      native RNG 下抽樣分布不變，但同一個 seed/nonce 抽出的序列會跟
      skip_zero=False 不同（被跳過的格子不再消耗亂數）。
    預設 False，保持原本 V3 的行為與 README 數字可重現。
    """

    assert probs.is_cuda
    assert probs.dtype == torch.float32
    assert probs.is_contiguous()
    assert temperature > 0

    if noise is not None:
        assert noise.device == probs.device
        assert noise.dtype == torch.float64
        assert noise.shape == probs.shape
        assert noise.is_contiguous()

    assert x.dtype == torch.int64
    assert x.device == probs.device
    assert x.is_contiguous()
    assert x.shape == probs.shape[:-1]

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
            x.data_ptr(),
            mask_index,
            rows, vocab, seed, nonce,
            temperature, int(bool(skip_zero)),
            stream.cuda_stream
        )

    if status != 0:
        raise RuntimeError(f"CUDA error: {status}")

    return output
