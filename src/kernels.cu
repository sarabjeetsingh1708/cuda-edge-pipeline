// Copyright 2026. GPU kernels for the batch edge-detection pipeline.
#include "kernels.h"

#include <cmath>

namespace edge {
namespace {

constexpr int kTile = 16;
constexpr int kRadius = 2;
constexpr int kSide = kTile + 2 * kRadius;
constexpr int kTaps = 2 * kRadius + 1;
static_assert(kTile * kTile == kBins, "one thread per histogram bin");

__constant__ float g_taps[kTaps];

// Cooperatively loads the (kTile + 2 * kRadius)^2 halo tile with edge
// clamping. Must be called by every thread of the block.
__device__ void LoadTile(const uint8_t* in, int width, int height,
                         uint8_t* tile) {
  const int x0 = blockIdx.x * kTile - kRadius;
  const int y0 = blockIdx.y * kTile - kRadius;
  for (int i = threadIdx.y * kTile + threadIdx.x; i < kSide * kSide;
       i += kTile * kTile) {
    const int x = min(max(x0 + i % kSide, 0), width - 1);
    const int y = min(max(y0 + i / kSide, 0), height - 1);
    tile[i] = in[y * width + x];
  }
  __syncthreads();
}

__global__ void BlurKernel(const uint8_t* in, uint8_t* out, int width,
                           int height) {
  __shared__ uint8_t tile[kSide * kSide];
  LoadTile(in, width, height, tile);
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  if (x >= width || y >= height) return;
  float acc = 0.f;
#pragma unroll
  for (int dy = 0; dy < kTaps; ++dy) {
#pragma unroll
    for (int dx = 0; dx < kTaps; ++dx) {
      acc += g_taps[dy] * g_taps[dx] *
             tile[(threadIdx.y + dy) * kSide + threadIdx.x + dx];
    }
  }
  out[y * width + x] = static_cast<uint8_t>(acc + 0.5f);
}

__global__ void SobelHistogramKernel(const uint8_t* in, uint8_t* out,
                                     unsigned* hist, int width, int height) {
  __shared__ uint8_t tile[kSide * kSide];
  __shared__ unsigned local_hist[kBins];
  const int bin = threadIdx.y * kTile + threadIdx.x;
  local_hist[bin] = 0;  // Made visible by the barrier inside LoadTile.
  LoadTile(in, width, height, tile);
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  if (x < width && y < height) {
    const uint8_t* p = tile + (threadIdx.y + kRadius) * kSide +
                       threadIdx.x + kRadius;
    const int gx = (p[-kSide + 1] + 2 * p[1] + p[kSide + 1]) -
                   (p[-kSide - 1] + 2 * p[-1] + p[kSide - 1]);
    const int gy = (p[kSide - 1] + 2 * p[kSide] + p[kSide + 1]) -
                   (p[-kSide - 1] + 2 * p[-kSide] + p[-kSide + 1]);
    const int mag = min(abs(gx) + abs(gy), 255);
    out[y * width + x] = static_cast<uint8_t>(mag);
    atomicAdd(&local_hist[mag], 1u);
  }
  __syncthreads();
  if (local_hist[bin] != 0) atomicAdd(&hist[bin], local_hist[bin]);
}

dim3 Grid(int width, int height) {
  return dim3((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
}

}  // namespace

void InitGaussian(float sigma) {
  float taps[kTaps], sum = 0.f;
  for (int i = 0; i < kTaps; ++i) {
    const float d = static_cast<float>(i - kRadius);
    sum += taps[i] = std::exp(-d * d / (2.f * sigma * sigma));
  }
  for (float& t : taps) t /= sum;
  CHECK_CUDA(cudaMemcpyToSymbol(g_taps, taps, sizeof(taps)));
}

void LaunchBlur(const uint8_t* in, uint8_t* out, int width, int height,
                cudaStream_t stream) {
  BlurKernel<<<Grid(width, height), dim3(kTile, kTile), 0, stream>>>(
      in, out, width, height);
  CHECK_CUDA(cudaGetLastError());
}

void LaunchSobelHistogram(const uint8_t* in, uint8_t* out, unsigned* hist,
                          int width, int height, cudaStream_t stream) {
  SobelHistogramKernel<<<Grid(width, height), dim3(kTile, kTile), 0,
                         stream>>>(in, out, hist, width, height);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace edge
