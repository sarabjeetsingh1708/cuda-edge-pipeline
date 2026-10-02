# Submission text

## Code Project Description (paste into Coursera)

I built a batch image-processing pipeline in CUDA C++ that takes a folder of images and, for each one, runs a 5x5 Gaussian blur, a Sobel edge detector and a 256-bin histogram of edge strength, then writes the edge map and a CSV of statistics. The goal was to show GPU usage on many pieces of data in a single execution, so the program accepts a directory plus CLI flags (streams, sigma, threshold) and processes hundreds of images with several CUDA streams and pinned memory so transfers and kernels of different images overlap. The blur and Sobel kernels use shared-memory tiles with a halo and edge clamping, the Gaussian taps live in constant memory, and the histogram uses per-block shared-memory bins merged with atomicAdd to avoid heavy global-memory contention. The hardest parts were getting the halo loading right for images whose size is not a multiple of the block size, making sure every thread reaches __syncthreads(), and getting real overlap between streams (which required pinned host memory and stream-ordered allocation). Comparing 1 stream against 4 streams in run.log showed the benefit of overlap, and the pipeline is mostly memory-bound, which matches what the course taught about tiling. Next steps: fuse blur and Sobel into one kernel, add non-maximum suppression and hysteresis (full Canny), use NPP/nvJPEG to decode on the GPU, and profile with Nsight Compute.

## Presentation outline (5-10 min, record screen + voice)

1. 0:00-1:00 Goal and why batch processing fits the GPU.
2. 1:00-3:00 Pipeline and kernels: show kernels.cu (tile + halo, constant memory, shared histogram).
3. 3:00-4:30 Streams and pinned memory: show Submit() in main.cu.
4. 4:30-6:30 Live demo: `./run.sh`, show log, summary.csv and 2-3 before/after images.
5. 6:30-8:00 Challenges and lessons learned, 1 vs N stream timing from run.log.
6. 8:00-9:00 Next steps (Canny, nvJPEG, Nsight profiling).
