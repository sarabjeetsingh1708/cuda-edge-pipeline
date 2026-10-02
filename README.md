# CUDA Batch Edge-Detection Pipeline

GPU Specialization capstone project. Processes a whole directory of images
through a three-stage CUDA pipeline and reports edge statistics per image:

1. **Gaussian blur (5x5)**: shared-memory tiled kernel, taps in constant memory.
2. **Sobel gradient magnitude**: reads the blurred image from a shared-memory halo tile.
3. **Histogram of edge strength**: per-block shared-memory histogram merged
   into a global one with `atomicAdd`.

Images are spread over N CUDA streams with pinned host memory, so uploads,
kernels and downloads of different images overlap (`cudaMallocAsync` is used
for stream-ordered device allocation).

## Requirements

- NVIDIA GPU + CUDA Toolkit >= 11.6 (`nvcc`, `nvidia-smi`)
- Python 3 with `numpy` and `Pillow` (only to generate synthetic test data)
- No GPU locally? Use Google Colab (Runtime > T4 GPU), then
  `!git clone <this repo> && cd <repo> && bash run.sh`

`stb_image.h` / `stb_image_write.h` (public domain) are vendored in `third_party/`.

## Build and run

```bash
make                      # builds bin/edge_pipeline  (ARCH=sm_75 to override)
./run.sh                  # build + run on data/ (synthetic if empty) + pack proof
./run.sh path/to/images 8 # custom dataset, 8 streams
```

Direct CLI use:

```bash
./bin/edge_pipeline --input_dir data --output_dir results/output \
    --streams 4 --sigma 1.4 --threshold 64 --max_images 0
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--input_dir` | folder with png/jpg/bmp/tga images | `data` |
| `--output_dir` | edge maps (`*_edges.png`) and `summary.csv` | `results` |
| `--streams` | concurrent CUDA streams | 4 |
| `--sigma` | Gaussian sigma | 1.4 |
| `--threshold` | magnitude >= threshold counts as an edge pixel | 64 |
| `--max_images` | limit images processed (0 = all) | 0 |

## Output

- `results/output/<name>_edges.png`: edge map for each input
- `results/output/summary.csv`: per-image width, height, mean edge strength, edge pixel count and percentage
- `results/run.log`, `results/gpu_info.txt`: console log, GPU details, and a 1-stream vs N-stream timing comparison
- `proof_of_execution.tar.gz`: all of the above plus sample inputs (created by `run.sh`)

## Data

`scripts/make_synthetic.py` generates 200 grayscale images (256-2048 px) by
default. Real data such as the USC SIPI database also works: drop the files
into `data/` (convert TIFF to PNG first).

## Layout

```
src/kernels.h, kernels.cu   CUDA kernels + launch wrappers
src/main.cu                 CLI, image I/O, stream scheduling, CSV output
scripts/make_synthetic.py   test data generator
Makefile, run.sh            build / run / package proof
third_party/                stb image headers
```
