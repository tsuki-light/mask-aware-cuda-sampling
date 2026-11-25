#include <stdint.h>
#include <cuda_runtime_api.h>

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
);

extern "C" int sampling_forward(
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
    return sampling_launch(
        probs, noise, output, rows, vocab,
        seed, nonce, temperature, stream
    );
}
