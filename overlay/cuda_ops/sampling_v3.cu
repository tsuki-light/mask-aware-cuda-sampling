#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

constexpr int THREADS = 256;
constexpr int PARTS = 8;

__device__ __forceinline__
void warp_reduce(double& best, int& idx) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double v = __shfl_down_sync(
            0xffffffff, best, offset);
        int i = __shfl_down_sync(
            0xffffffff, idx, offset);

        if (v > best || (v == best && i < idx)) {
            best = v;
            idx = i;
        }
    }
}

__global__
void sampling_partial(
    const float* probs,
    const double* noise,
    double* partial_values,
    int* partial_indices,
    const int64_t* x,
    int64_t mask_index,
    int vocab,
    uint64_t seed,
    uint64_t nonce,
    double temperature,
    int skip_zero
) {
    int row = blockIdx.x;

    if (x[row] != mask_index)
        return;

    int part = blockIdx.y;
    int tid = threadIdx.x;

    int64_t base = (int64_t)row * vocab;

    curandStatePhilox4_32_10_t rng;

    if (!noise) {
        uint64_t subsequence =
            ((uint64_t)row * PARTS + part)
            * THREADS + tid;

        curand_init(
            seed, subsequence,
            nonce * (2ULL * vocab), &rng);
    }

    // skip_zero:
    //   p = 0 的格子分數必為 0 / g = 0，贏不了任何正分，
    //   所以直接跳過，省掉 log / div 這段 FP64 運算。
    //   起點設成 (0, 0)：整列都是 0 時仍回傳 index 0，
    //   跟 torch.argmax 一樣。
    // skip_zero = 0 時行為與原本 V3 逐 bit 相同。
    double best = skip_zero ? 0.0 : -INFINITY;
    int best_idx = skip_zero ? 0 : vocab;

    for (int i = part * THREADS + tid;
         i < vocab;
         i += PARTS * THREADS) {

        float p = probs[base + i];

        if (skip_zero && p == 0.0f)
            continue;

        double u = noise
            ? noise[base + i]
            : curand_uniform_double(&rng);

        if (u == 1.0)
            u = nextafter(1.0, 0.0);

        double g = -log(u);

        if (temperature != 1.0)
            g = pow(g, temperature);

        double score = (double)p / g;

        if (score > best ||
            (score == best && i < best_idx)) {
            best = score;
            best_idx = i;
        }
    }

    warp_reduce(best, best_idx);

    __shared__ double values[8];
    __shared__ int indices[8];

    int lane = tid % 32;
    int warp = tid / 32;

    if (lane == 0) {
        values[warp] = best;
        indices[warp] = best_idx;
    }

    __syncthreads();

    if (warp == 0) {
        best = lane < 8 ? values[lane] : -INFINITY;
        best_idx = lane < 8 ? indices[lane] : vocab;

        warp_reduce(best, best_idx);

        if (lane == 0) {
            int pos = row * PARTS + part;
            partial_values[pos] = best;
            partial_indices[pos] = best_idx;
        }
    }
}

__global__
void sampling_finalize(
    const double* values,
    const int* indices,
    int64_t* output,
    const int64_t* x,
    int64_t mask_index,
    int vocab
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;

    if (x[row] != mask_index) {
        if (lane == 0)
            output[row] = x[row];
        return;
    }

    double best = lane < PARTS
        ? values[row * PARTS + lane]
        : -INFINITY;

    int idx = lane < PARTS
        ? indices[row * PARTS + lane]
        : vocab;

    warp_reduce(best, idx);

    if (lane == 0)
        output[row] = idx;
}

extern "C" int sampling_forward_v3(
    const float* probs,
    const double* noise,
    int64_t* output,
    double* partial_values,
    int* partial_indices,
    const int64_t* x,
    int64_t mask_index,
    int rows,
    int vocab,
    uint64_t seed,
    uint64_t nonce,
    double temperature,
    int skip_zero,
    cudaStream_t stream
) {
    sampling_partial<<<
        dim3(rows, PARTS), THREADS, 0, stream
    >>>(
        probs, noise,
        partial_values, partial_indices,
        x, mask_index, vocab, seed, nonce, temperature,
        skip_zero
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return (int)err;

    sampling_finalize<<<rows, 32, 0, stream>>>(
        partial_values, partial_indices,
        output, x, mask_index, vocab
    );

    return (int)cudaGetLastError();
}
