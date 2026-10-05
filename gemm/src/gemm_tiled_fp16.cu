// gemm_tiled_fp16.cu
// Shared-memory tiled GEMM, FP16 in/out, FP32 accumulation, C = alpha*A*B + beta*C.
// The tiled kernel and solve() below are the submitted version, unchanged.
// The driver at the bottom benchmarks it against a naive kernel with identical
// semantics (same dtypes, same epilogue, same 32x32 launch shape), so tiling is
// the only thing that differs between the two timings.
//
// Build: nvcc -O3 -arch=sm_89 gemm_tiled_fp16.cu -o gemm_tiled_fp16
// Run:   ./gemm_tiled_fp16 [sizes...]        (default: 256 512 1000 1024 2048 4096)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32
#define WARP_SIZE 32
#define COARSE 4


__global__ void matrix_multiplication_kernel_simp( const half * A, const half * B, half * C, int M, int N, int K, float alpha, float beta){
    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float Pvalue1 = 0;
    int iters = (K + TILE_SIZE -1)/TILE_SIZE;
    for (int i = 0; i < iters; i++) {
        int acol = TILE_SIZE * i + tx;
        int brow = i * TILE_SIZE + ty;

        if (row < M && acol < K) {
            Ashared[ty][tx] = __half2float(A[K * row + acol]);
        } else {
            Ashared[ty][tx] = 0.0f;
        }
        if (brow < K && col < N) {
            Bshared[ty][tx] = __half2float(B[brow * N + col]);
        } else {
            Bshared[ty][tx] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            Pvalue1 += Ashared[ty][k] * Bshared[k][tx];
        }
        __syncthreads();
    }
    int idx = row * N + col;

    if (row < M && col < N) {
        half out0 = __float2half(Pvalue1 * alpha + beta * __half2float(C[idx]));
        C[idx] = out0;
    }

}
// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {

        int dimRows = (M + TILE_SIZE - 1) / TILE_SIZE;
        int dimCols = (N + TILE_SIZE - 1) / TILE_SIZE;
        dim3 dimGrid( dimCols, dimRows, 1);
        dim3 dimBlock( TILE_SIZE, TILE_SIZE, 1);
        matrix_multiplication_kernel_simp<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
        cudaDeviceSynchronize();

                      }

// ---------------------------------------------------------------------------
// Benchmark driver (not part of the submission)
// ---------------------------------------------------------------------------

#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e__ = (call);                                               \
        if (e__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e__));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Naive baseline with the same contract as the tiled kernel: one thread per C
// element, FP16 loads straight from global memory, FP32 accumulate, same epilogue.
__global__ void gemm_naive_fp16(const half* A, const half* B, half* C, int M, int N, int K,
                                float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    int idx = row * N + col;
    C[idx] = __float2half(acc * alpha + beta * __half2float(C[idx]));
}

static float time_kernel(bool tiled, const half* dA, const half* dB, half* dC, int M, int N,
                         int K, int warmup, int iters, float* min_ms) {
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    auto launch = [&]() {
        if (tiled) matrix_multiplication_kernel_simp<<<grid, block>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f);
        else       gemm_naive_fp16<<<grid, block>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f);
    };
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    float total = 0.0f, best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(s));
        launch();
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    *min_ms = best;
    return total / iters;
}

static void run_size(int S, FILE* csv) {
    const int M = S, N = S, K = S;
    std::vector<half> hA((size_t)M * K), hB((size_t)K * N), hCn((size_t)M * N), hCt((size_t)M * N);
    srand(1234 + S);
    for (auto& v : hA) v = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    for (auto& v : hB) v = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);

    half *dA, *dB, *dC;
    size_t bA = sizeof(half) * hA.size(), bB = sizeof(half) * hB.size(), bC = sizeof(half) * hCn.size();
    CUDA_CHECK(cudaMalloc(&dA, bA));
    CUDA_CHECK(cudaMalloc(&dB, bB));
    CUDA_CHECK(cudaMalloc(&dC, bC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bB, cudaMemcpyHostToDevice));

    const int warmup = 2, iters = 5;
    float naive_min, tiled_min;
    CUDA_CHECK(cudaMemset(dC, 0, bC));
    float naive_ms = time_kernel(false, dA, dB, dC, M, N, K, warmup, iters, &naive_min);
    CUDA_CHECK(cudaMemcpy(hCn.data(), dC, bC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(dC, 0, bC));
    float tiled_ms = time_kernel(true, dA, dB, dC, M, N, K, warmup, iters, &tiled_min);
    CUDA_CHECK(cudaMemcpy(hCt.data(), dC, bC, cudaMemcpyDeviceToHost));

    // tiled vs naive: same math, different summation order, both rounded to half
    double max_tn = 0.0;
    for (size_t i = 0; i < hCt.size(); ++i) {
        double d = fabs((double)__half2float(hCt[i]) - (double)__half2float(hCn[i]));
        if (d > max_tn) max_tn = d;
    }

    // CPU fp32 reference (same half inputs) for sizes small enough to finish quickly
    double max_ref = -1.0, max_mag = 0.0;
    if (S <= 1024) {
        std::vector<float> fA(hA.size()), fB(hB.size());
        for (size_t i = 0; i < hA.size(); ++i) fA[i] = __half2float(hA[i]);
        for (size_t i = 0; i < hB.size(); ++i) fB[i] = __half2float(hB[i]);
        std::vector<float> ref((size_t)M * N, 0.0f);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = fA[(size_t)i * K + k];
                const float* brow = &fB[(size_t)k * N];
                float* crow = &ref[(size_t)i * N];
                for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
            }
        max_ref = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double d = fabs((double)__half2float(hCt[i]) - (double)ref[i]);
            if (d > max_ref) max_ref = d;
            if (fabs(ref[i]) > max_mag) max_mag = fabs(ref[i]);
        }
    }

    double flops = 2.0 * M * N * K;
    double naive_gf = flops / (naive_ms / 1e3) / 1e9;
    double tiled_gf = flops / (tiled_ms / 1e3) / 1e9;
    printf("S=%-5d naive %9.4f ms %8.1f GFLOPS | tiled %9.4f ms %8.1f GFLOPS | speedup %5.2fx | "
           "max|tiled-naive| %.4g",
           S, naive_ms, naive_gf, tiled_ms, tiled_gf, naive_ms / tiled_ms, max_tn);
    if (max_ref >= 0) printf(" | max|tiled-cpu_fp32| %.4g (max|C| %.1f)", max_ref, max_mag);
    printf("\n");
    fprintf(csv, "%d,%.5f,%.5f,%.2f,%.5f,%.5f,%.2f,%.3f,%.6g,%.6g\n", S, naive_ms, naive_min,
            naive_gf, tiled_ms, tiled_min, tiled_gf, naive_ms / tiled_ms, max_tn,
            max_ref >= 0 ? max_ref : NAN);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// The tiled kernel always reads C, even when beta == 0. BLAS convention is to
// not read C at all when beta == 0, because 0 * NaN = NaN. This fills C with NaN
// first and counts how many NaNs come out with alpha=1, beta=0.
static void beta_zero_nan_check() {
    const int S = 64;
    std::vector<half> hA(S * S), hB(S * S), hC(S * S);
    for (auto& v : hA) v = __float2half(0.5f);
    for (auto& v : hB) v = __float2half(0.5f);
    half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half) * S * S));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(half) * S * S));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(half) * S * S));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half) * S * S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half) * S * S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0xFF, sizeof(half) * S * S));  // 0xFFFF = half NaN
    solve(dA, dB, dC, S, S, S, 1.0f, 0.0f);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(half) * S * S, cudaMemcpyDeviceToHost));
    int nans = 0;
    for (auto& v : hC) if (isnan(__half2float(v))) ++nans;
    printf("\nbeta=0 with NaN-filled C: %d / %d outputs are NaN (expected 0 if C were ignored; "
           "correct value is %.1f)\n", nans, S * S, 0.25f * S);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1000, 1024, 2048, 4096};

    FILE* csv = fopen("../results/tiled_fp16_vs_naive.csv", "w");
    if (!csv) csv = fopen("tiled_fp16_vs_naive.csv", "w");
    fprintf(csv, "S,naive_mean_ms,naive_min_ms,naive_gflops,tiled_mean_ms,tiled_min_ms,tiled_gflops,"
                 "speedup,max_abs_tiled_vs_naive,max_abs_tiled_vs_cpu_fp32\n");
    for (int S : sizes) run_size(S, csv);
    fclose(csv);

    beta_zero_nan_check();
    return 0;
}
