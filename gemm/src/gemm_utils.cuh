// gemm_utils.cuh
// Shared helpers for every GEMM kernel version in this lab (naive, tiled, and
// whatever comes after) -- not naive-kernel-specific, so it lives outside
// gemm_naive.cu on purpose.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Row-major host-side matrix: M rows, N cols, contiguous, element (i,j) at
// data[i*N + j]. Matches PyTorch/NumPy's default C-contiguous layout, so the
// Python correctness/benchmark harnesses can hand this binary raw buffers with
// no transpose surprises.
struct HostMatrix {
    int rows, cols;
    float* data;

    HostMatrix(int r, int c) : rows(r), cols(c) {
        data = static_cast<float*>(std::malloc(sizeof(float) * r * c));
    }
    ~HostMatrix() { std::free(data); }

    HostMatrix(const HostMatrix&) = delete;
    HostMatrix& operator=(const HostMatrix&) = delete;
};

// Reads raw float32 binary (row-major) from a file written by the Python side
// (numpy's .tofile()), rather than generating random data independently in two
// places -- so the CUDA binary and the Python reference are always comparing
// against the literal same input, not two "should be equivalent" randoms.
inline void read_matrix_bin(const char* path, HostMatrix& m) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s\n", path);
        std::exit(1);
    }
    size_t n = static_cast<size_t>(m.rows) * m.cols;
    size_t got = std::fread(m.data, sizeof(float), n, f);
    std::fclose(f);
    if (got != n) {
        std::fprintf(stderr, "Short read on %s: got %zu of %zu floats\n", path, got, n);
        std::exit(1);
    }
}

inline void write_matrix_bin(const char* path, const HostMatrix& m) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for write\n", path);
        std::exit(1);
    }
    size_t n = static_cast<size_t>(m.rows) * m.cols;
    std::fwrite(m.data, sizeof(float), n, f);
    std::fclose(f);
}

// Timing helper: CUDA-event-based, matching how latency is measured throughout
// the rest of this repo (scripts/build_and_bench.py's benchmark()) -- one
// consistent methodology across the TensorRT side and the CUDA side of the lab.
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start, stream)); }
    float end_ms(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
