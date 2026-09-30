// gemm_naive.cu
// Naive GEMM: one thread per output element, no shared memory, no tiling.
// Design this implements: ../docs/NAIVE_KERNEL_SPEC.md -- read that first if
// anything here looks unmotivated.
//
// CLI:
//   ./gemm_naive --M 512 --N 512 --K 512
//       benchmark mode: random A/B, prints "M=.. N=.. K=.. mean_ms=.. min_ms=.. gflops=.."
//   ./gemm_naive --M 64 --N 64 --K 64 --a-file a.bin --b-file b.bin --c-out c.bin
//       correctness mode: A/B read from raw float32 row-major binaries
//       (numpy .tofile() format), C written the same way for the Python side
//       to compare against a NumPy reference.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

static void fill_random(HostMatrix& m, unsigned seed) {
    std::srand(seed);
    size_t n = static_cast<size_t>(m.rows) * m.cols;
    for (size_t i = 0; i < n; ++i) {
        m.data[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
    }
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    const char *a_file = nullptr, *b_file = nullptr, *c_out = nullptr;
    int warmup = 2, iters = 5;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--M") && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--N") && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--K") && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--a-file") && i + 1 < argc) a_file = argv[++i];
        else if (!std::strcmp(argv[i], "--b-file") && i + 1 < argc) b_file = argv[++i];
        else if (!std::strcmp(argv[i], "--c-out") && i + 1 < argc) c_out = argv[++i];
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    HostMatrix hA(M, K), hB(K, N), hC(M, N);
    if (a_file) read_matrix_bin(a_file, hA); else fill_random(hA, 1234);
    if (b_file) read_matrix_bin(b_file, hB); else fill_random(hB, 5678);

    float *dA, *dB, *dC;
    size_t bytesA = sizeof(float) * static_cast<size_t>(M) * K;
    size_t bytesB = sizeof(float) * static_cast<size_t>(K) * N;
    size_t bytesC = sizeof(float) * static_cast<size_t>(M) * N;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data, bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warmup: not timed. Matches this repo's TensorRT-side convention
    // (build_and_bench.py) of warming up before any measurement counts.
    for (int i = 0; i < warmup; ++i) {
        gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    GpuTimer timer;
    float total_ms = 0.0f, min_ms = 1e30f;
    for (int i = 0; i < iters; ++i) {
        timer.begin();
        gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        float ms = timer.end_ms();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hC.data, dC, bytesC, cudaMemcpyDeviceToHost));
    if (c_out) write_matrix_bin(c_out, hC);

    double mean_ms = total_ms / iters;
    double gflops = (2.0 * M * N * K) / (mean_ms / 1000.0) / 1e9;
    std::printf("M=%d N=%d K=%d mean_ms=%.5f min_ms=%.5f gflops=%.3f\n",
                M, N, K, mean_ms, min_ms, gflops);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
