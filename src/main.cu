// Copyright 2026. Batch GPU edge detection: Gaussian blur -> Sobel ->
// histogram, over a directory of images using multiple CUDA streams.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"
#include "kernels.h"

namespace fs = std::filesystem;

namespace {

constexpr char kUsage[] =
    "Usage: edge_pipeline [options]\n"
    "  --input_dir DIR    directory of png/jpg/bmp/tga images (default data)\n"
    "  --output_dir DIR   where edge maps + summary.csv go (default results)\n"
    "  --streams N        concurrent CUDA streams (default 4)\n"
    "  --sigma F          Gaussian sigma (default 1.4)\n"
    "  --threshold N      edge-pixel magnitude threshold 0-255 (default 64)\n"
    "  --max_images N     process at most N images, 0 = all (default 0)\n";

// One image in flight: pinned host buffers, device buffers, bound stream.
struct Job {
  std::string name;
  int width = 0, height = 0;
  uint8_t *h_in = nullptr, *h_out = nullptr;
  uint8_t *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
  unsigned *h_hist = nullptr, *d_hist = nullptr;
  cudaStream_t stream = nullptr;
};

std::map<std::string, std::string> ParseArgs(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || i + 1 >= argc) {
      std::cout << kUsage;
      std::exit(std::strcmp(argv[i], "--help") == 0 ? 0 : 1);
    }
    args[argv[i] + 2] = argv[i + 1];
    ++i;
  }
  return args;
}

// Loads `path` as 8-bit grayscale and enqueues the whole pipeline on the
// job's stream. Returns false if the image cannot be decoded.
bool Submit(const fs::path& path, Job* job) {
  int channels = 0;
  uint8_t* pixels =
      stbi_load(path.string().c_str(), &job->width, &job->height, &channels, 1);
  if (pixels == nullptr) return false;
  const int w = job->width, h = job->height;
  const size_t bytes = static_cast<size_t>(w) * h;
  job->name = path.filename().string();
  CHECK_CUDA(cudaMallocHost(&job->h_in, bytes));
  CHECK_CUDA(cudaMallocHost(&job->h_out, bytes));
  CHECK_CUDA(cudaMallocHost(&job->h_hist, edge::kBins * sizeof(unsigned)));
  std::memcpy(job->h_in, pixels, bytes);
  stbi_image_free(pixels);

  cudaStream_t s = job->stream;
  CHECK_CUDA(cudaMallocAsync(&job->d_in, bytes, s));
  CHECK_CUDA(cudaMallocAsync(&job->d_tmp, bytes, s));
  CHECK_CUDA(cudaMallocAsync(&job->d_out, bytes, s));
  CHECK_CUDA(cudaMallocAsync(&job->d_hist, edge::kBins * sizeof(unsigned), s));
  CHECK_CUDA(cudaMemsetAsync(job->d_hist, 0, edge::kBins * sizeof(unsigned), s));
  CHECK_CUDA(cudaMemcpyAsync(job->d_in, job->h_in, bytes,
                             cudaMemcpyHostToDevice, s));
  edge::LaunchBlur(job->d_in, job->d_tmp, w, h, s);
  edge::LaunchSobelHistogram(job->d_tmp, job->d_out, job->d_hist, w, h, s);
  CHECK_CUDA(cudaMemcpyAsync(job->h_out, job->d_out, bytes,
                             cudaMemcpyDeviceToHost, s));
  CHECK_CUDA(cudaMemcpyAsync(job->h_hist, job->d_hist,
                             edge::kBins * sizeof(unsigned),
                             cudaMemcpyDeviceToHost, s));
  return true;
}

// Waits for the job, writes its edge map, appends a CSV row, frees buffers.
void Finish(Job* job, const fs::path& out_dir, int threshold,
            std::ofstream* csv) {
  CHECK_CUDA(cudaStreamSynchronize(job->stream));
  const double total = static_cast<double>(job->width) * job->height;
  double weighted = 0, edges = 0;
  for (int b = 0; b < edge::kBins; ++b) {
    weighted += static_cast<double>(b) * job->h_hist[b];
    if (b >= threshold) edges += job->h_hist[b];
  }
  const std::string out_name = fs::path(job->name).stem().string() + "_edges.png";
  stbi_write_png((out_dir / out_name).string().c_str(), job->width, job->height,
                 1, job->h_out, job->width);
  *csv << job->name << ',' << job->width << ',' << job->height << ','
       << weighted / total << ',' << edges << ',' << 100.0 * edges / total
       << '\n';
  CHECK_CUDA(cudaFreeHost(job->h_in));
  CHECK_CUDA(cudaFreeHost(job->h_out));
  CHECK_CUDA(cudaFreeHost(job->h_hist));
  CHECK_CUDA(cudaFreeAsync(job->d_in, job->stream));
  CHECK_CUDA(cudaFreeAsync(job->d_tmp, job->stream));
  CHECK_CUDA(cudaFreeAsync(job->d_out, job->stream));
  CHECK_CUDA(cudaFreeAsync(job->d_hist, job->stream));
}

}  // namespace

int main(int argc, char** argv) {
  auto args = ParseArgs(argc, argv);
  auto get = [&](const char* key, const char* fallback) {
    return args.count(key) ? args[key] : std::string(fallback);
  };
  const fs::path in_dir = get("input_dir", "data");
  const fs::path out_dir = get("output_dir", "results");
  const int num_streams = std::max(1, std::stoi(get("streams", "4")));
  const float sigma = std::stof(get("sigma", "1.4"));
  const int threshold = std::stoi(get("threshold", "64"));
  const size_t max_images = std::stoul(get("max_images", "0"));

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(in_dir)) {
    const std::string ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
        ext == ".tga") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  if (max_images > 0 && files.size() > max_images) files.resize(max_images);
  if (files.empty()) {
    std::cerr << "No images found in " << in_dir << "\n";
    return 1;
  }
  fs::create_directories(out_dir);

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " | images: " << files.size()
            << " | streams: " << num_streams << " | sigma: " << sigma << "\n";
  edge::InitGaussian(sigma);

  std::vector<cudaStream_t> streams(num_streams);
  for (auto& s : streams) CHECK_CUDA(cudaStreamCreate(&s));
  std::ofstream csv(out_dir / "summary.csv");
  csv << "image,width,height,mean_edge_strength,edge_pixels,edge_percent\n";

  const auto start = std::chrono::steady_clock::now();
  double megapixels = 0;
  size_t done = 0;
  for (size_t i = 0; i < files.size(); i += num_streams) {
    std::vector<Job> jobs(std::min<size_t>(num_streams, files.size() - i));
    std::vector<bool> ok(jobs.size());
    for (size_t k = 0; k < jobs.size(); ++k) {
      jobs[k].stream = streams[k];
      ok[k] = Submit(files[i + k], &jobs[k]);
      if (!ok[k]) std::cerr << "Skipping unreadable " << files[i + k] << "\n";
    }
    for (size_t k = 0; k < jobs.size(); ++k) {
      if (!ok[k]) continue;
      megapixels += 1e-6 * jobs[k].width * jobs[k].height;
      Finish(&jobs[k], out_dir, threshold, &csv);
      ++done;
    }
    std::cout << "processed " << done << "/" << files.size() << "\r"
              << std::flush;
  }
  for (auto& s : streams) CHECK_CUDA(cudaStreamSynchronize(s));
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - start).count();
  std::cout << "\nDone: " << done << " images, " << megapixels << " MPix in "
            << secs << " s (" << done / secs << " img/s, " << megapixels / secs
            << " MPix/s)\n";
  for (auto& s : streams) CHECK_CUDA(cudaStreamDestroy(s));
  return 0;
}
