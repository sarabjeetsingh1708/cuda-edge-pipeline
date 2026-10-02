// Copyright 2026. GPU kernels for the batch edge-detection pipeline.
#ifndef CUDA_EDGE_PIPELINE_SRC_KERNELS_H_
#define CUDA_EDGE_PIPELINE_SRC_KERNELS_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                   cudaGetErrorString(err_), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

namespace edge {

constexpr int kBins = 256;

// Uploads the 1-D Gaussian taps (5-tap, separable weights) to constant memory.
void InitGaussian(float sigma);

// 5x5 Gaussian blur, shared-memory tiled. `in` and `out` are device buffers
// of width * height bytes.
void LaunchBlur(const uint8_t* in, uint8_t* out, int width, int height,
                cudaStream_t stream);

// Sobel gradient magnitude (|gx| + |gy|, clamped to 255) written to `out`,
// plus a 256-bin histogram of the magnitudes accumulated into `hist`
// (device memory, must be zeroed by the caller).
void LaunchSobelHistogram(const uint8_t* in, uint8_t* out, unsigned* hist,
                          int width, int height, cudaStream_t stream);

}  // namespace edge

#endif  // CUDA_EDGE_PIPELINE_SRC_KERNELS_H_
