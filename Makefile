# Usage: make [ARCH=sm_75]   (default: native GPU, needs CUDA >= 11.6)
NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)
BIN := bin/edge_pipeline
SRCS := src/main.cu src/kernels.cu

all: $(BIN)

$(BIN): $(SRCS) src/kernels.h
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

run: $(BIN)
	./run.sh

clean:
	rm -rf bin results

.PHONY: all run clean
