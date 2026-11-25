#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

constexpr int THREADS = 256;

__device__ __forceinline__
void warp_reduce(double& best, int& idx) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double v = __shfl_down_sync(0xffffffff, best, offset);
        int i = __shfl_down_sync(0xffffffff, idx, offset);

        if (v > best || (v == best && i < idx)) {
            best = v;
            idx = i;
        }
    }
}

__global__
void sampling_kernel(
    const float* probs,
    const double* noise,
    int64_t* output,
    int vocab,
    uint64_t seed,
    uint64_t nonce,
    double temperature
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    int64_t base = (int64_t)row * vocab;

    curandStatePhilox4_32_10_t rng;

    if (noise == nullptr) {
        curand_init(
            seed,
            (uint64_t)row * blockDim.x + tid,
            nonce * (2ULL * vocab),
            &rng
        );
    }

    double best = -INFINITY;
    int best_idx = vocab;

    // Each thread processes part of the vocabulary.
    for (int i = tid; i < vocab; i += blockDim.x) {
        double u = noise
            ? noise[base + i]
            : curand_uniform_double(&rng);

        // cuRAND produces (0, 1]; exclude 1.
        if (u == 1.0)
            u = nextafter(1.0, 0.0);

        double g = -log(u);

        if (temperature != 1.0)
            g = pow(g, temperature);

        double score = (double)probs[base + i] / g;

        if (score > best ||
            (score == best && i < best_idx)) {
            best = score;
            best_idx = i;
        }
    }

    // Stage 1: Warp reduction.
    warp_reduce(best, best_idx);

    __shared__ double warp_best[8];
    __shared__ int warp_idx[8];

    if (lane == 0) {
        warp_best[warp] = best;
        warp_idx[warp] = best_idx;
    }

    __syncthreads();

    // Stage 2: Block reduction.
    if (warp == 0) {
        best = lane < 8 ? warp_best[lane] : -INFINITY;
        best_idx = lane < 8 ? warp_idx[lane] : vocab;

        warp_reduce(best, best_idx);

        if (lane == 0)
            output[row] = best_idx;
    }
}


extern "C" int sampling_launch(
    const float* probs,
    const double* noise,
    int64_t* output,
    int rows,
    int vocab,
    uint64_t seed,
    uint64_t nonce,
    double temperature,
    cudaStream_t stream
) {
    sampling_kernel<<<rows, THREADS, 0, stream>>>(
        probs, noise, output, vocab,
        seed, nonce, temperature
    );

    return (int)cudaGetLastError();
}
