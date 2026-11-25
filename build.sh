#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/overlay"

ARCH="${CUDA_ARCH:-sm_89}"

mkdir -p build

nvcc -O3 -arch="$ARCH" \
  --shared -Xcompiler -fPIC \
  cuda_ops/binding.cpp cuda_ops/sampling.cu \
  -o build/libsampling.so

nvcc -O3 -arch="$ARCH" \
  --shared -Xcompiler -fPIC \
  cuda_ops/sampling_v2.cu \
  -o build/libsampling_v2.so

nvcc -O3 -arch="$ARCH" \
  --shared -Xcompiler -fPIC \
  cuda_ops/sampling_v3.cu \
  -o build/libsampling_v3.so

echo "Build complete."
